from new_basis_llp_qaoa.qaoa import QAOA
from scipy.linalg import expm

from new_basis_llp_qaoa.qaoa.parameters import Parameters


class CYPStateVector:
    def __init__(self, ha, hb, p, n):
        self.problem_hamiltonian = ha
        self.mixer_hamiltonian = hb
        self.p = p
        self.qubits_number = n
        self.current_parameters = None
        self.current_state = None

    def unitary_a(self, gamma):
        ha = self.problem_hamiltonian
        return expm(-1j * gamma * ha)

    def unitary_b(self, beta):
        hb = self.mixer_hamiltonian
        return expm(-1j * beta * hb)

    def evolve(self, parameters: Parameters) -> None:
        gammas = parameters.gamma_list
        betas = parameters.beta_list
        for gamma, beta in zip(gammas, betas):
            ua = self.unitary_a(gamma)
            ub = self.unitary_b(beta)
            self.current_state = ub @ ua @ self.current_state

    def run_optimizer(self):
        pass

    def compute_energy(self):
        pass
