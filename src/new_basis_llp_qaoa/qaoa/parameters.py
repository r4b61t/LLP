from dataclasses import dataclass

import numpy as np


@dataclass
class Parameters:
    gamma_list: np.ndarray
    beta_list: np.ndarray
