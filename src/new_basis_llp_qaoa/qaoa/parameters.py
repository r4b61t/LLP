from dataclasses import dataclass

import numpy as np
from scipy.optimize import OptimizeResult


@dataclass
class Parameters:
    gamma_list: np.ndarray
    beta_list: np.ndarray

    @classmethod
    def from_optimization_result(cls, res: OptimizeResult):
        params = res.x
        p = int(len(params)/2)
        betas = params[:p]
        gammas = params[p:]
        return Parameters(
            gamma_list=gammas,
            beta_list=betas
        )
