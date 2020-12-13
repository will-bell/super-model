import numpy as np
from typing import Tuple
from common.functions import distance


class Potential:

    minimum: np.ndarray

    zeta: float

    quadratic_radius: float

    def __init__(self, minimum: np.ndarray, zeta: float = 1., quadratic_radius: float = 1.):
        self._minimum, self._zeta, self._quadratic_radius = self._verify(minimum, zeta, quadratic_radius)

    @staticmethod
    def _verify(minimum: np.ndarray, zeta: float, quadratic_radius: float) -> Tuple[np.ndarray, float, float]:
        assert minimum.size == 2, 'Goal must be a two-dimensional point'
        assert zeta > 0, 'Zeta must be a value greater than 0'
        assert quadratic_radius > 0, 'Eta must be a value greater than 0'
        return minimum.copy(), zeta, quadratic_radius

    def evaluate_potential(self, point: np.ndarray) -> float:
        d2min = distance(point, self._minimum)
        if d2min <= self._quadratic_radius:
            value = .5 * self._zeta * d2min ** 2
        else:
            value = self._quadratic_radius * self._zeta * d2min - .5 * self._zeta * self._quadratic_radius ** 2

        return value

    def evaluate_potential_gradient(self, point: np.ndarray) -> np.ndarray:
        d2min = distance(point, self._minimum)
        if d2min <= self._quadratic_radius:
            value = self._zeta * (point - self._minimum)
        else:
            value = self._quadratic_radius * self._zeta * (point - self._minimum) / d2min

        return value
