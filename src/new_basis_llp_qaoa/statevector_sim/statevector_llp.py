import math

from functools import cached_property

from typing import Optional

import numpy as np
from scipy.sparse.linalg import expm
from scipy.optimize import minimize
from scipy.sparse import csc_matrix, kron, identity, dia_matrix
from tqdm import tqdm
from time import time
import logging
from new_basis_llp_qaoa.qaoa.parameters import Parameters

N_OP = csc_matrix([[0, 0], [0, 1]])
PAULI_X = csc_matrix([[0, 1], [1, 0]])
XNX_OP = PAULI_X @ N_OP @ PAULI_X
PAULI_I = np.identity(2)


def get_energy(psi, ha) -> float:
    bra_psi = psi.conj().transpose()
    ket_psi = psi

    return bra_psi @ (ha @ ket_psi)


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


def nu(action_num, max_n):
    res = N_OP if 0 in binary_1(action_num) else XNX_OP
    for i in range(1, max_n):
        if i in binary_1(action_num):
            res = kron(N_OP, res)
        else:
            res = kron(XNX_OP, res)
    return res


def nu_ij(i, j, max_n, loanees):
    before = i * max_n
    after = (loanees - i - 1) * max_n
    before = identity(2 ** before)
    after = identity(2 ** after)
    res = kron(nu(j, max_n), before)
    res = kron(after, res)
    return res


def unitary_a(ha, gamma):
    ha = dia_matrix(ha)
    shape = ha.shape
    dia = -1j * gamma * ha.diagonal()
    dia = np.exp(dia)
    res = dia_matrix((dia, 0), shape=shape)
    return csc_matrix(res)


def unitary_b(hb, beta):
    return expm(-1j * beta * hb)


def unitary_b_taylor(term_lists, beta):
    def fact(i):
        res = math.factorial(i)
        return res
    try:
        return np.sum((-1j* beta) ** i * term / fact(i) for i, term in enumerate(term_lists))
    except Exception as e:
        print(beta)
        raise e


class StateVectorLLP:
    def __init__(self, Qs, As, epsilon, p, initial_state="ones"):
        self.p = p
        self.Qs = Qs
        self.As = As
        self.epsilon = epsilon
        loanee_number, action_number = Qs.shape
        self.loanee_number = loanee_number
        self.action_number = action_number
        self.qubits_number = get_qubits_number(loanee_number, action_number)
        self.current_parameters = None
        self.initial_state = initial_state
        self.psi: Optional[np.ndarray] = None

    @cached_property
    def ha(self):
        logging.info("Caching HA.")
        return self.construct_problem_hamiltonian()

    @cached_property
    def hb(self):
        logging.info("Caching HB.")
        return self.construct_mixer_hamiltonian()

    @cached_property
    def hb_taylor_terms(self, terms: int = 12) -> list:
        hb = self.hb
        return [hb ** i for i in tqdm(range(terms), "Calculating taylor series terms for HB.")]

    def construct_problem_hamiltonian(self):
        e = self.epsilon
        profit_term = self.profit_term
        welfare_term = self.welfare_term
        return -(1 - e) * profit_term - e * welfare_term

    @property
    def max_n(self):
        return qubits_number_for_number_of_actions(self.action_number)

    @property
    def profit_term(self):
        n = self.qubits_number
        res = csc_matrix((2 ** n, 2 ** n))
        for i in tqdm(range(self.loanee_number), "Calculating profit term."):
            for j in range(2 ** self.max_n):
                extra = np.ones((self.loanee_number, 2 ** self.max_n - self.action_number)) * -99  # Create energy hills
                h_ij = np.hstack((self.Qs, extra))[i, j]
                res += h_ij * nu_ij(i, j, self.max_n, loanees=self.loanee_number)
        return res

    @property
    def welfare_term(self):
        n = self.qubits_number
        res = csc_matrix((2 ** n, 2 ** n))
        for i in tqdm(range(self.loanee_number), "Calculating welfare term."):
            for i_prime in range(i + 1, self.loanee_number):
                first_factor = identity(2 ** n) - nu_ij(i, 0, self.max_n, loanees=self.loanee_number)
                second_factor = identity(2 ** n) - nu_ij(i_prime, 0, self.max_n, loanees=self.loanee_number)
                res += self.As[i, i_prime] * (first_factor @ second_factor)
        return res

    def construct_mixer_hamiltonian(self):
        n = self.qubits_number
        res = kron(identity(2 ** (n - 1)), PAULI_X)
        for q in tqdm(range(1, n), "Calculating HB."):
            before = q
            after = n - q - 1
            before = identity(2 ** before)
            after = identity(2 ** after)
            current_x = kron(PAULI_X, before)
            current_x = kron(after, current_x)
            res += current_x
        return res

    def evolve(self, parameters: Parameters) -> None:
        gammas = parameters.gamma_list
        betas = parameters.beta_list
        hb_taylor_terms = self.hb_taylor_terms
        ha = self.ha
        if self.psi is None:
            self.psi = self.get_initial_state(self.qubits_number, self.initial_state)
        for gamma, beta in zip(gammas, betas):
            ua = unitary_a(ha, gamma)
            ub = unitary_b_taylor(hb_taylor_terms, beta)
            self.psi = ub @ ua @ self.psi

    def get_energy(self) -> float:
        bra_psi = self.psi.conj().transpose()
        ket_psi = self.psi

        return bra_psi @ (self.ha @ ket_psi)

    def optimized(self, maxiter=None, method="COBYLA"):
        p = self.p
        params = self.run_optimizer(maxiter, method).x
        betas = params[:p]
        gammas = params[p:]
        params = Parameters(
            gamma_list=gammas,
            beta_list=betas
        )
        logging.info("Evolving psi")
        self.evolve(params)

    def run_optimizer(self, maxiter=None, method="COBYLA"):
        p = self.p
        ha = self.ha
        hb_taylor_terms = self.hb_taylor_terms
        beta_bounds = [(-np.pi, np.pi)]
        gamma_bound = [(-np.pi, np.pi)]
        bounds = beta_bounds * p + gamma_bound * p
        if self.psi is None:
            self.psi = self.get_initial_state(self.qubits_number, self.initial_state)

        def cost(v):
            psi = self.psi.copy()
            betas = v[:p]
            gammas = v[p:]
            for gamma, beta in zip(gammas, betas):
                if beta > 4:
                    raise Exception
                ua = unitary_a(ha, gamma)
                ub = unitary_b_taylor(hb_taylor_terms, beta)
                # ub = unitary_b(hb, beta)
                psi = ub @ ua @ psi
            return get_energy(psi, ha).real

        def callback(xk):
            pbar.update(1)

        with tqdm(total=maxiter, desc="Optimizing") as pbar:
            optimizer_result = minimize(
                cost,
                np.ones(2 * p),
                method=method,
                options={'maxiter': maxiter},
                bounds=bounds,
                callback=callback
            )

        return optimizer_result

    @property
    def probabilities(self):
        return np.absolute(self.psi) ** 2

    def get_energy2(self):
        return self.probabilities.T @ self.ha.diagonal()

    @staticmethod
    def get_initial_state(n: int, state_str="ones"):
        logging.info("Getting initial state.")
        if state_str == "ones":
            state = np.ones(2 ** n)
            state = state / np.sqrt(2 ** n)
        else:
            state = np.zeros(2 ** n)
            state[0] = 1
        return state

    def get_str_from_index(self, ind):
        q_number = get_qubits_number(self.loanee_number, self.action_number)
        action_bit_len = int(qubits_number_for_action(self.action_number))
        return get_str(ind, q_number, action_bit_len)


    def get_cost_from_str(self, state):
        c = 0
        h = (1-self.epsilon) * self.Qs
        J = self.epsilon * self.As
        for i in range(len(state)):
            action = int(state[i])
            if action >= self.action_number:
                return 0
            c -= h[i, action]
            for ii in range(i):
                if action == 0 and int(state[ii]) == 0:
                    c -= J[i, ii]
        return c

def get_str(ind, q_number, action_bit_len):
    binary = bin(ind)[2:]
    bitstring = '0' * (q_number - len(binary)) + str(binary)
    actions_for_each_loanee = [bitstring[i:i + action_bit_len] for i in range(0, len(bitstring), action_bit_len)]
    actions_for_each_loanee = [int(action_bin, 2) for action_bin in actions_for_each_loanee]
    res = ''.join(map(str, (actions_for_each_loanee[::-1])))
    return res
