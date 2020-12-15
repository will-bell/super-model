from typing import Tuple, Union

import numpy as np
import torch

from supermodel.functions import distance


class Potential:

    minimum: np.ndarray

    zeta: float

    quadratic_radius: float

    def __init__(self, minimum: Union[np.ndarray, torch.Tensor], zeta: float = 1., quadratic_radius: float = 1.):
        self._np_minimum, self._tr_minimum, self._zeta, self._quadratic_radius = self._verify(minimum, zeta, quadratic_radius)

    @staticmethod
    def _verify(minimum: np.ndarray, zeta: float, quadratic_radius: float) -> Tuple[np.ndarray, torch.Tensor, float, float]:
        if isinstance(minimum, np.ndarray):
            assert minimum.size == 2, 'Goal must be a two-dimensional point'
            np_minimum = minimum.copy()
            tr_minimum = torch.Tensor(minimum.copy())
        elif isinstance(minimum, torch.Tensor):
            assert minimum.numel() == 2, 'Goal must be a two-dimensional point'
            tr_minimum = minimum.clone()
            np_minimum = minimum.clone().numpy()
        else:
            raise ValueError('minimum must be an np.ndarray or torch.Tensor')
        assert zeta > 0, 'Zeta must be a value greater than 0'
        assert quadratic_radius >= 0, 'Quadratic radius must be a value greater than or equal to 0'
        return np_minimum, tr_minimum, zeta, quadratic_radius

    def evaluate_potential(self, point: Union[np.ndarray, torch.Tensor]) -> Union[float, torch.Tensor]:
        if isinstance(point, torch.Tensor):
            device = point.device
            minimum = self._tr_minimum.to(device)
        elif isinstance(point, np.ndarray):
            minimum = self._np_minimum
        else:
            raise ValueError('point must be either an np.ndarray or torch.Tensor')

        d2min = distance(point, minimum)
        if d2min < self._quadratic_radius:
            value = .5 * self._zeta * d2min ** 2
        else:
            if self._quadratic_radius < 1e-6:
                value = self._zeta * d2min
            else:
                value = self._quadratic_radius * self._zeta * d2min - .5 * self._zeta * self._quadratic_radius ** 2

        return value

    def evaluate_potential_gradient(self, point: np.ndarray) -> np.ndarray:
        d2min = distance(point, self._np_minimum)
        if d2min <= self._quadratic_radius:
            value = self._zeta * (point - self._np_minimum)
        else:
            value = self._quadratic_radius * self._zeta * (point - self._np_minimum) / d2min

        return value
