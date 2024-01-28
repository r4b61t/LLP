from scipy.optimize import minimize
from typing import Optional

import math

import logging

from functools import cached_property

from tqdm import tqdm

import numpy as np
from collections import namedtuple

from new_basis_llp_qaoa.statevector_sim.statevector_llp import get_str

Bound = namedtuple('Bound', ['beta', 'gamma'])


class StateVectorLLPV2:
    N_OP = np.array([[0, 0], [0, 1]])
    PAULI_X = np.array([[0, 1], [1, 0]])
    XNX_OP = PAULI_X @ N_OP @ PAULI_X
    PAULI_I = np.identity(2)

    def __init__(self, h: np.ndarray, A: np.ndarray, epsilon: float, p):
        """
         h[i,j] (Qs): expected net profit that the bank would get if action j is taken to loanee i.
         A[i, i'] (As): an association matrix, defined as the averaged transaction between loanees i and i′.
         epsilon: The hyperparameter in [0, 1) tunes the competition between the expected return to the bank and the
         financial welfare in the network of loanees.
         p: Number of layers during QAOA.
        """
        self.p = p
        self.h = h
        self.A = A
        self.epsilon = epsilon
        loanee_number, action_number = h.shape
        self.loanee_number: int = loanee_number
        self.action_number: int = action_number
        self.action_bit_len: int = int(np.ceil(np.log2(self.action_number)))
        self.qubits_number: int = self.action_bit_len * self.loanee_number
        self.bounds = Bound(beta=(-np.pi / 2, np.pi / 2), gamma=(-np.pi / 2, np.pi / 2))
        self.dt: Optional[int] = None
        self.taylor_terms: Optional[int] = None
        self.psi: Optional[np.ndarray] = None

    def profit_term(self) -> np.ndarray:
        n = self.qubits_number
        res = np.zeros((2 ** n, 2 ** n))
        for i in tqdm(range(self.loanee_number), "Calculating profit term."):
            for j in range(2 ** self.action_bit_len):
                extra = np.ones(
                    (self.loanee_number, 2 ** self.action_bit_len - self.action_number)) * -99  # Create energy hills
                h_ij = np.hstack((self.h, extra))[i, j]
                res += h_ij * self.nu_ij(i, j)
        return res

    def nu_ij(self, i: int, j: int) -> np.ndarray:
        before = i * self.action_bit_len
        after = (self.loanee_number - i - 1) * self.action_bit_len
        before = np.identity(2 ** before)
        after = np.identity(2 ** after)
        res = np.kron(self.nu(j), before)
        res = np.kron(after, res)
        return res

    def nu(self, action: int) -> np.ndarray:
        def binary_1(j: int):
            bit_string = bin(j)[2:][::-1]
            indexed_bits = enumerate(bit_string)
            return set(index for index, bit in indexed_bits if bit == '1')

        res = self.N_OP if 0 in binary_1(action) else self.XNX_OP
        for i in range(1, self.action_bit_len):
            if i in binary_1(action):
                res = np.kron(self.N_OP, res)
            else:
                res = np.kron(self.XNX_OP, res)
        return res

    def welfare_term(self) -> np.ndarray:
        n = self.qubits_number
        res = np.zeros((2 ** n, 2 ** n))
        iden = np.identity(2 ** n)
        for i in tqdm(range(self.loanee_number), "Calculating welfare term."):
            nu_i0 = self.nu_ij(i, 0)
            for i_prime in range(i + 1, self.loanee_number):
                nu_i_prime0 = self.nu_ij(i_prime, 0)
                first_factor = iden - nu_i0
                second_factor = iden - nu_i_prime0
                res += self.A[i, i_prime] * (first_factor @ second_factor)
        return res

    def construct_problem_hamiltonian(self) -> np.ndarray:
        e = self.epsilon
        profit_term = self.profit_term()
        welfare_term = self.welfare_term()
        return -(1 - e) * profit_term - e * welfare_term

    @cached_property
    def ha(self):
        logging.info("Caching HA.")
        return self.construct_problem_hamiltonian()

    def unitary_a(self, gamma: float):
        dia = -1j * gamma * self.ha.diagonal()
        dia = np.exp(dia)
        return np.diag(dia)

    def construct_mixer_hamiltonian(self) -> np.ndarray:
        n = self.qubits_number
        res = np.kron(np.identity(2 ** (n - 1)), self.PAULI_X)
        for q in tqdm(range(1, n), "Calculating HB."):
            before = q
            after = n - q - 1
            before = np.identity(2 ** before)
            after = np.identity(2 ** after)
            current_x = np.kron(self.PAULI_X, before)
            current_x = np.kron(after, current_x)
            res += current_x
        return res

    @cached_property
    def hb(self) -> np.ndarray:
        logging.info("Caching HB.")
        res = self.construct_mixer_hamiltonian()
        return res

    def unitary_b(self, beta: float, dt=0.1, taylor_terms=8) -> np.ndarray:
        """ Uses trotterization.

        cache ub with beta in [-1.6, 1.6] at 0.1 intervals
        given beta separate it into 2 parts, beta = beta_coarse + beta_fine
        beta_coarse, beta_fine = separate_beta(beta)
        beta_fine = beta - beta_course
        ub = ub(beta_course) * ub(beta_fine)
        """

        self.dt = dt
        self.taylor_terms = taylor_terms
        beta_coarse, beta_fine = self.separate_beta(beta)
        # logging.warning("accessing ub coarse cache")
        ub_integer = self.ub_integer_cache[beta_coarse]
        # logging.warning("calculating ub_fine")
        if np.abs(beta_fine) < 1e-2:
            return ub_integer
        ub_decimal = self.unitary_b_taylor(beta_fine)
        # logging.warning("calculating ub")
        return ub_integer @ ub_decimal

    def separate_beta(self, beta) -> tuple[int, float]:
        dt = self.dt
        c, f = beta.__divmod__(dt)
        c = int(c)
        lower_bound, upper_bound = self.bounds.beta
        upper_bound = math.ceil(upper_bound / dt)
        lower_bound = math.floor(lower_bound / dt)
        if c > upper_bound:
            c = upper_bound
        if c < lower_bound:
            c = lower_bound
        if f > dt / 2:
            c = c + 1
            f = f - dt
        return c, f

    @cached_property
    def ub_integer_cache(self) -> dict[int, np.ndarray]:
        # logging.info("Calculating ub_small")
        ub_small = self.unitary_b_taylor(self.dt)
        _, upper_bound = self.bounds.beta
        res = dict()
        ub_small_inv = np.linalg.inv(ub_small)
        for i in tqdm(
                range(math.ceil(upper_bound / self.dt) + 1),
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

    def unitary_b_taylor(self, beta: float) -> np.ndarray:

        n = len(self.hb_taylor_terms)
        factorials = np.array([math.factorial(i) for i in range(n)])
        powers = (-1j * beta) ** np.arange(n)

        res = powers * np.array(self.hb_taylor_terms).T / factorials
        return np.sum(res, axis=-1)

    @cached_property
    def hb_taylor_terms(self) -> list[np.ndarray]:
        hb = self.hb
        res = [np.identity(2 ** self.qubits_number)]
        for i in tqdm(range(1, self.taylor_terms), "Calculating taylor series terms for HB."):
            term = hb @ res[i - 1]
            res.append(term)
        return res

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

    def run_optimizer(self, maxiter=None, method="COBYLA", initial_state='ones'):
        p = self.p
        bounds = [self.bounds.beta] * p + [self.bounds.gamma] * p
        if self.psi is None:
            self.psi = self.get_initial_state(self.qubits_number, initial_state)

        def cost(v):
            psi = self.psi.copy()
            betas = v[:p]
            gammas = v[p:]
            for gamma, beta in zip(gammas, betas):
                ua = self.unitary_a(gamma)
                ub = self.unitary_b(beta)
                psi = ub @ ua @ psi
            energy = self.get_energy(psi)
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
            )

        return optimizer_result

    def get_energy(self, psi: Optional[np.ndarray] = None) -> float:
        if psi is None:
            psi = self.psi
        return np.dot(self.probabilities(psi).T, self.ha.diagonal())

    def get_energy_true(self):
        """Actual way of getting energy."""
        bra_psi = self.psi.conj().transpose()
        ket_psi = self.psi

        return bra_psi @ (self.ha @ ket_psi)

    def probabilities(self, psi: Optional[np.ndarray] = None) -> np.ndarray:
        if psi is None:
            psi = self.psi
        return np.absolute(psi) ** 2

    def evolve(self, parameters: list[float]) -> None:
        logging.info("Evolving psi")
        p = int(len(parameters)/2)
        gammas = parameters[p:]
        betas = parameters[:p]
        if self.psi is None:
            self.psi = self.get_initial_state(self.qubits_number)
        for gamma, beta in zip(gammas, betas):
            ua = self.unitary_a(gamma)
            ub = self.unitary_b(beta)
            self.psi = ua @ self.psi
            self.psi = ub @ self.psi

    def optimized(self, maxiter=None, method="COBYLA"):
        params = self.run_optimizer(maxiter, method).x
        self.evolve(params)

    def get_str_from_index(self, ind):
        return get_str(ind, self.qubits_number, self.action_bit_len)

    def get_cost_from_str(self, state):
        c = 0
        h = (1 - self.epsilon) * self.h
        J = self.epsilon * self.a
        for i in range(len(state)):
            action = int(state[i])
            if action >= self.action_number:
                return 0
            c -= h[i, action]
            for ii in range(i):
                if action == 0 and int(state[ii]) == 0:
                    c -= J[i, ii]
        return c
