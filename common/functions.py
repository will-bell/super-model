import numpy as np
from common.sphere_obstacle import SphereObstacle
import sys

EPS = sys.float_info.epsilon


def distance(point1: np.ndarray, point2: np.ndarray) -> float:
    return np.linalg.norm(point1 - point2)
