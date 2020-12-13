from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from common.functions import distance, EPS


class SphereObstacle:

    def __init__(self, center: Tuple[float, float], radius: float):
        self.center = np.asarray(center)
        self.radius = radius

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
            lim = 1.1 * self.radius
            ax.set_xlim([-lim, lim])
            ax.set_ylim([-lim, lim])
            ax.set_aspect('equal', 'box')

        circle_plot = plt.Circle(tuple(self.center), self.radius, color='r')
        ax.add_artist(circle_plot)


def sphere_distance(point: np.ndarray, sphere: SphereObstacle) -> float:
    return distance(point, sphere.center) - sphere.radius


def sphere_distance_gradient(point: np.ndarray, sphere: SphereObstacle) -> np.ndarray:
    rel_pos = point - sphere.center
    if sphere.radius < 0:
        rel_pos = -rel_pos
    den = np.linalg.norm(rel_pos)
    if den < EPS:
        gradient = np.array([0., 0.])
    else:
        gradient = rel_pos / den

    return gradient
