from abc import ABC, abstractmethod


class QAOA(ABC):
    @property
    @abstractmethod
    def problem_hamiltonian(self):
        pass

    @property
    @abstractmethod
    def mixer_hamiltonian(self):
        pass

    @property
    @abstractmethod
    def p(self):
        pass

    @property
    @abstractmethod
    def qubits_number(self):
        pass

    @abstractmethod
    def evolve(self, parameter):
        pass

    @abstractmethod
    def run_optimizer(self):
        pass

    @abstractmethod
    def compute_energy(self):
        pass
