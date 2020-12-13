import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt


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