from typing import Optional

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from new_basis_llp_qaoa.qaoa.parameters import Parameters
from new_basis_llp_qaoa.statevector_sim import CYPStateVector

N_OP = np.array([[0, 0], [0, 1]])
PAULI_X = np.array([[0, 1], [1, 0]])
XNX_OP = PAULI_X @ N_OP @ PAULI_X
PAULI_I = np.identity(2)


def qubits_number_for_action(action):
    return np.ceil(np.log2(action))


def get_qubits_number(loanee, action):
    return int(qubits_number_for_action(action) * loanee)


def binary_1(j):
    bit_string = bin(j)[2:][::-1]
    indexed_bits = enumerate(bit_string)
    return set(index for index, bit in indexed_bits if bit == '1')


def binary_0(j):
    bit_string = bin(j)[2:][::-1]
    indexed_bits = enumerate(bit_string)
    return set(index for index, bit in indexed_bits if bit == '0')


def qubits_number_for_number_of_actions(action):
    return int(np.ceil(np.log2(action)))


def nu(action_num, max_number_of_qubits):
    res = N_OP if 0 in binary_1(action_num) else XNX_OP

    for i in range(1, max_number_of_qubits):
        if i in binary_1(action_num):
            res = np.kron(N_OP, res)
        else:
            res = np.kron(XNX_OP, res)
    return res


def nu_ij(i, j, max_n, loanees):
    before = i * max_n
    after = (loanees - i - 1) * max_n
    before = np.identity(2 ** before)
    after = np.identity(2 ** after)
    res = np.kron(nu(j, max_n), before)
    res = np.kron(after, res)
    return res


class StateVectorLLP:
    def __init__(self, Qs, As, epsilon, initial_state):
        self.Qs = Qs
        self.As = As
        self.epsilon = epsilon
        loanee_number, action_number = Qs.shape
        self.loanee_number = loanee_number
        self.action_number = action_number
        self.qubits_number = get_qubits_number(loanee_number, action_number)
        self.ha = self.construct_problem_hamiltonian()
        self.hb = self.construct_mixer_hamiltonian()
        self.current_parameters = None
        self.current_state :Optional[np.ndarray] = self.get_initial_state(self.qubits_number, initial_state)

    def construct_problem_hamiltonian(self):
        e = self.epsilon
        return -(1 - e) * self.profit_term - e * self.welfare_term

    @property
    def max_n(self):
        return qubits_number_for_number_of_actions(self.action_number)

    @property
    def profit_term(self):
        n = self.qubits_number
        res = np.zeros((2 ** n, 2 ** n))
        for i in range(self.loanee_number):
            for j in range(self.action_number):
                h_ij = self.Qs[i, j]
                res += h_ij * nu_ij(i, j, self.max_n, loanees=self.loanee_number)
        return res

    @property
    def welfare_term(self):
        n = self.qubits_number
        res = np.zeros((2 ** n, 2 ** n))
        for i in range(self.loanee_number):
            for i_prime in range(i + 1, self.loanee_number):
                first_factor = np.identity(2 ** n) - nu_ij(i, 0, self.max_n, loanees=self.loanee_number)
                second_factor = np.identity(2 ** n) - nu_ij(i_prime, 0, self.max_n, loanees=self.loanee_number)
                res += self.As[i, i_prime] * (first_factor @ second_factor)
        return res

    def construct_mixer_hamiltonian(self):
        n = self.qubits_number
        res = np.kron(PAULI_X, np.identity(2**(n-1)))
        for q in range(1, n):
            before = q
            after = n - q -1
            before = np.identity(2**before)
            after = np.identity(2**after)
            current_x = np.kron(before, PAULI_X)
            current_x = np.kron(current_x, after)
            res += current_x
        return res


    def unitary_a(self, gamma):
        ha = self.ha
        return expm(-1j * gamma * ha)

    def unitary_b(self, beta):
        hb = self.hb
        return expm(-1j * beta * hb)

    def evolve(self, parameters: Parameters) -> None:
        gammas = parameters.gamma_list
        betas = parameters.beta_list
        for gamma, beta in zip(gammas, betas):
            ua = self.unitary_a(gamma)
            ub = self.unitary_b(beta)
            self.current_state = ub @ ua @ self.current_state

    def get_energy(self) -> float:
        bra_psi = self.current_state.conj().transpose()
        ket_psi = self.current_state

        return bra_psi @ (self.ha @ ket_psi)

    def run_optimizer(self, p):
        def cost(v):
            svllp = StateVectorLLP(
                self.Qs, self.As, self.epsilon, "ones"
            )
            beta = v[:p]
            gamma = v[p:]
            parameter = Parameters(
                gamma_list=gamma,
                beta_list=beta,
            )
            svllp.evolve(parameter)
            res = svllp.get_energy()
            print(res)
            return res

        optimizer_result = minimize(cost, np.ones(2*p))
        return optimizer_result

    @property
    def probabilities(self):
        return np.absolute(self.current_state)**2

    def get_energy2(self):
        return self.probabilities.T @ self.ha.diagonal()

    def get_initial_state(self, n: int, state_str = "ones"):
        if state_str == "ones":
            state = np.ones(2**n)
            state = state / np.sqrt(2 ** n)
        else:
            state = np.zeros(2**n)
            state[0] = 1
        return state


