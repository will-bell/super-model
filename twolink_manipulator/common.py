import numpy as np
from twolink_manipulator.sphere_obstacle import SphereObstacle
import sys

EPS = sys.float_info.epsilon


def distance(point1: np.ndarray, point2: np.ndarray) -> float:
    return np.linalg.norm(point1 - point2)


def sphere_distance(point: np.ndarray, sphere: SphereObstacle) -> float:
    return distance(point, sphere.center) - sphere.radius


def sphere_distance_gradient(point: np.ndarray, sphere: SphereObstacle) -> float:
    rel_pos = point - sphere.center
    if sphere.radius < 0:
        rel_pos = -rel_pos
    den = np.linalg.norm(rel_pos)
    if den < EPS:
        gradient = np.array([0., 0.])
    else:
        gradient = rel_pos / den

    return gradient
