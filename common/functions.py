import numpy as np
import sys
from typing import Union
import torch

EPS = sys.float_info.epsilon


def distance(point1: Union[np.ndarray, torch.Tensor], point2: Union[np.ndarray, torch.Tensor]) -> float:
    if isinstance(point1, np.ndarray) and isinstance(point2, np.ndarray):
        return np.linalg.norm(point1 - point2)
    elif isinstance(point1, torch.Tensor) and isinstance(point2, torch.Tensor):
        return torch.linalg.norm(point1 - point2)
    else:
        raise ValueError('point1 and point2 must both be an np.ndarray or torch.Tensor')
