from numba import njit, complex128, float64
from numbers import Complex

import math

from functools import cached_property

from typing import Optional

import numpy as np
from scipy.sparse.linalg import expm
import scipy
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


def seperate_beta(beta, interval):
    c, f = beta.__divmod__(interval)
    c = int(c)
    if f > interval / 2:
        c = c + 1
        f = f - interval
    # if np.abs(f) < 1e-2:
    #     f = 0
    assert np.isclose(c * interval + f, beta, atol=2e-2), (c * interval + f, beta)
    return c, f


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


def unitary_a(ha, gamma, use_sparse):
    if not use_sparse:
        dia = -1j * gamma * ha.diagonal()
        dia = np.exp(dia)
        return np.diag(dia)

    ha = dia_matrix(ha)
    shape = ha.shape
    dia = -1j * gamma * ha.diagonal()
    dia = np.exp(dia)
    res = dia_matrix((dia, 0), shape=shape)
    return csc_matrix(res)


def fast_ub(hb, beta):
    return scipy.linalg.expm(-1j * beta * hb)


def unitary_b(hb, beta, use_sparse):
    if use_sparse:
        return expm(-1j * beta * hb)
    return fast_ub(hb.toarray(), beta)


def unitary_b_taylor(term_lists: list[csc_matrix], beta: float, use_sparse: bool):
    n = len(term_lists)
    if not use_sparse:
        term_lists = np.array([term.toarray() for term in term_lists]).T
    try:
        factorials = np.array([math.factorial(i) for i in range(n)])
        powers = (-1j * beta) ** np.arange(n)

        res = powers * term_lists / factorials
        return np.sum(res)
    except Exception as e:
        logging.warning(beta)
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
        self.use_sparse = self.qubits_number > 12
        self.psi: Optional[np.ndarray] = None
        self.terms = 10
        self.interval = 0.1

    @cached_property
    def ha(self):
        logging.info("Caching HA.")
        return self.construct_problem_hamiltonian()

    @cached_property
    def hb(self) -> csc_matrix:
        logging.info("Caching HB.")
        res = self.construct_mixer_hamiltonian()
        return res

    @cached_property
    def ub_integer_cache(self):
        # logging.info("Calculating ub_small")
        ub_small = unitary_b_taylor(self.hb_taylor_terms, self.interval, True).toarray()

        res = dict()
        ub_small_inv = np.linalg.inv(ub_small)
        for i in tqdm(
                range(math.ceil(1.6 // self.interval) + 1),
                "caching small ub"
        ):
            if i == 0:
                res[i] = np.identity(len(ub_small))
            elif i == 1:
                res[i] = ub_small
                res[-i] = ub_small_inv
            else:
                res[i] = res[i - 1] @ ub_small
                res[-i] = res[-i + 1] @ ub_small_inv

        return res

    def unitary_b_approx(self, beta: float, interval=0.1, terms=10):
        """
        cache ub with beta in [-1.6, 1.6] at 0.1 intervals
        given beta separate it into 2 parts, beta = beta_coarse + beta_fine
        beta_coarse, beta_fine = separate_beta(beta)
        beta_fine = beta - beta_course
        ub = ub(beta_course) * ub(beta_fine)
        """
        self.terms = terms
        self.interval = interval
        beta_coarse, beta_fine = seperate_beta(beta, interval)
        # logging.warning("accessing ub coarse cache")
        ub_integer = self.ub_integer_cache[beta_coarse]
        # logging.warning("calcing ub_fine")
        ub_decimal = unitary_b_taylor(self.hb_taylor_terms, beta_fine, True).toarray()
        # logging.warning("calcing ub")
        return ub_integer @ ub_decimal

    @cached_property
    def hb_taylor_terms(self) -> list[csc_matrix]:
        terms = self.terms
        hb = self.hb
        res = [identity(2 ** self.qubits_number)]
        for i in tqdm(range(1, terms), "Calculating taylor series terms for HB."):
            term = hb @ res[i - 1]
            res.append(term)
        return res

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
        if self.use_sparse:
            return res
        return res.toarray()

    @property
    def welfare_term(self):
        n = self.qubits_number
        res = csc_matrix((2 ** n, 2 ** n))
        iden = identity(2 ** n)
        for i in tqdm(range(self.loanee_number), "Calculating welfare term."):
            nu_i0 = nu_ij(i, 0, self.max_n, loanees=self.loanee_number)
            for i_prime in range(i + 1, self.loanee_number):
                nu_iprime0 = nu_ij(i_prime, 0, self.max_n, loanees=self.loanee_number)
                first_factor = iden - nu_i0
                second_factor = iden - nu_iprime0
                res += self.As[i, i_prime] * (first_factor @ second_factor)
        if self.use_sparse:
            return res
        return res.toarray()

    def construct_mixer_hamiltonian(self) -> csc_matrix:
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
        return csc_matrix(res)

    def evolve(self, parameters: Parameters) -> None:
        gammas = parameters.gamma_list
        betas = parameters.beta_list
        ha = self.ha
        if self.psi is None:
            self.psi = self.get_initial_state(self.qubits_number, self.initial_state)
        for gamma, beta in zip(gammas, betas):
            ua = unitary_a(ha, gamma, self.use_sparse)
            ub = self.unitary_b_approx(beta)
            print(sum(self.probabilities))
            self.psi = ua @ self.psi
            print(sum(self.probabilities))
            self.psi = ub @ self.psi
            print(sum(self.probabilities))

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

    def run_optimizer(self, maxiter=None, method="COBYLA", **kwargs):
        p = self.p
        ha = self.ha
        beta_bounds = [(-np.pi / 2, np.pi / 2)]
        gamma_bound = [(-np.pi / 2, np.pi / 2)]
        bounds = beta_bounds * p + gamma_bound * p
        if self.psi is None:
            self.psi = self.get_initial_state(self.qubits_number, self.initial_state)

        def cost(v):
            psi = self.psi.copy()
            betas = v[:p]
            gammas = v[p:]
            for gamma, beta in zip(gammas, betas):
                ua = unitary_a(ha, gamma, self.use_sparse)
                ub = self.unitary_b_approx(beta)
                psi = ub @ ua @ psi
            energy = get_energy(psi, ha).real
            print(gammas,'\n', betas,'\n', energy)
            return energy

        def callback(xk):
            pbar.update(1)

        with tqdm(total=maxiter, desc="Optimizing") as pbar:
            optimizer_result = minimize(
                cost,
                np.zeros(2 * p),
                method=method,
                options={
                    'maxiter': maxiter,
                    'tol': 1e-2,
                    'catol': 1e-5
                },
                bounds=bounds,
                callback=callback,
                *kwargs
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
        h = (1 - self.epsilon) * self.Qs
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
