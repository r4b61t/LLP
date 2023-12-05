from src.ibmq_utils import bit_repr, if_valid_state,  convert_bitstring
import numpy as np
from scipy.linalg import expm
from dataclasses import dataclass
from typing import List, Union


@dataclass
class CYPStatevectorSim:
    Qs: np.ndarray
    As: np.ndarray
    J: float
    Epsilon: float
    Parameters: Union[List, np.ndarray]
    InitialState: str

    @property
    def P(self):
        return int(len(self.Parameters)/2)

    @property
    def N(self):
        return self.Qs.shape[0]

    @property
    def M(self):
        return self.Qs.shape[1]

    @property
    def NQubits(self):
        return self.N * self.M

    def UA(self, gamma):
        HA = self.HA
        res = expm(-1j * gamma * HA)
        return res

    def UB(self, beta):
        HB = self.HB
        res = expm(-1j * beta * HB)
        return res

    @property
    def HA(self):
        N = self.N
        M = self.M
        e = self.Epsilon
        profit = self.Qs
        LLP = self.As

        profit_term = []
        for i in range(N):
            for j in range(M):
                _1 = ("I", [], float(profit[i][j]) * -(1-e) / 2)
                _2 = ("Z", [i*M + j], -float(profit[i][j]) * -(1-e) / 2)
                profit_term += [_1, _2]

        welfare_term = []
        for i in range(N):
            for i_prime in range(i+1, N):
                _1 = ("I", [], float(LLP[i][i_prime]) * -e/4)
                _2 = ("Z", [i*M], float(LLP[i][i_prime]) * - e/4)
                _3 = ("Z", [i_prime*M], float(LLP[i][i_prime]) * -e/4)
                _4 = ("ZZ", [i*M, i_prime*M], float(LLP[i][i_prime]) * -e/4)

                welfare_term += [_1, _2, _3, _4]

        profit_term = sum([self.construct_matrix(term) for term in profit_term])
        welfare_term = sum([self.construct_matrix(term) for term in welfare_term])
        res = profit_term + welfare_term

        return res

    @property
    def HB(self):
        N = self.N
        M = self.M
        J = self.J

        res = []
        for i in range(N):
            for j in range(M):
                j_plus1 = (j + 1) % M

                XX = ("XX", [i * M + j, i * M + j_plus1], -J / 2)
                YY = ("YY", [i * M + j, i * M + j_plus1], -J / 2)

                res += [XX, YY]

        ret = sum([self.construct_matrix(term) for term in res])
        return ret

    def construct_matrix(self, term: tuple):
        I = np.identity(2)
        X = np.array([
            [0, 1],
            [1, 0]
        ])
        Y = np.array([
            [0, -1j],
            [1j, 0]
        ])
        Z = np.array([
            [1, 0],
            [0, -1]
        ])
        res_name = ""
        name, targets, coeff = term
        res = 1
        if 'X' in name:
            pauli = X
            pauli_name = 'X'
        elif 'Y' in name:
            pauli = Y
            pauli_name = 'Y'
        elif 'Z' in name:
            pauli = Z
            pauli_name = 'Z'
        else:
            pauli = I
            pauli_name = 'I'
        for pos in range(self.NQubits):
            if pos in targets:
                res = np.kron(res, pauli)
                res_name += pauli_name
            else:
                res = np.kron(res, I)
                res_name += 'I'
        return coeff * res

    def construct_initial_state(self):
        N = self.N
        M = self.M
        state = self.InitialState

        states = np.zeros(2**(N*M))
        if state == "ones":
            for state_index in range(len(states)):
                if if_valid_state(bit_repr(state_index,N,M),M):
                    states[state_index] = 1
            return states/np.sqrt(M**N)
        elif state == "zeros":

            res = ['1' if i % M == 0 else '0' for i in range(M * N)]  # test this
            binary = ''.join(res[::-1])
            decimal = int(binary, 2)
            states[decimal] = 1  # 001001
            return states
        else:
            raise Exception

    def state_vector_sim_circuit(self):
        state_vector = self.construct_initial_state()
        for i in range(self.P):
            gamma = self.Parameters[2*i+1]
            beta = self.Parameters[2*i]
            state_vector = self.UB(beta) @ state_vector
            state_vector = self.UA(gamma) @ state_vector
        return state_vector

    def state_prob(self):
        state_vector = self.state_vector_sim_circuit()
        M = self.M
        N = self.N
        state_prob = {i: prob for i, prob in enumerate([np.abs(state) ** 2 for state in state_vector])}
        sorted_states = sorted(state_prob.items(), key=lambda x: -x[1])
        res = dict()
        for i, prob in sorted_states:
            if prob > 0.0001:
                res[convert_bitstring(bit_repr(i, N, M), M, N)] = prob
        return res
