import numpy as np
from typing import Tuple


class SphereObstacle:

    def __init__(self, center: Tuple[float, float], radius: float):
        self.center = np.asarray(center)
        self.radius = radius
