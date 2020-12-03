from typing import List

import cvxpy
import numpy as np
import matplotlib.pyplot as plt

from twolink_manipulator.common import sphere_distance, sphere_distance_gradient
from twolink_manipulator.potential import Potential
from twolink_manipulator.sphere_obstacle import SphereObstacle
from twolink_manipulator.twolink import TwoLink


def qp_min_effort(Aclf: np.ndarray, bclf: float, Acbf: np.ndarray, bcbf: np.ndarray, m: float) -> np.ndarray:
    u = cvxpy.Variable((2, 1))
    delta = cvxpy.Variable((1, 1))

    constraints = [Aclf @ u + bclf <= delta]
    if Acbf is not None:
        constraints.append(Acbf @ u + bcbf <= 0)

    problem = cvxpy.Problem(
        cvxpy.Minimize(cvxpy.sum_squares(u) + m*delta),
        constraints)
    problem.solve(solver=cvxpy.MOSEK)
    u_star = np.transpose(u.value)

    return u_star


def clfcbf_control(twolink: TwoLink, theta_eval: np.ndarray, potential: Potential, obstacles: List[SphereObstacle],
                   m: float) -> np.ndarray:
    p_eval = twolink.update_joints(tuple(theta_eval))
    Aclf = potential.evaluate_potential_gradient(p_eval) @ twolink.jacobian
    bclf = potential.evaluate_potential(p_eval)

    if len(obstacles):
        Acbf = []
        bcbf = []
        for obstacle in obstacles:
            Acbf.append(-sphere_distance_gradient(p_eval, obstacle) @ twolink.jacobian)
            bcbf.append(-sphere_distance(p_eval, obstacle))
        Acbf = np.vstack(Acbf)
        bcbf = np.vstack(bcbf)

    else:
        Acbf = bcbf = None

    u_star = qp_min_effort(Aclf, bclf, Acbf, bcbf, m)

    return u_star


def clfcbf_planner(twolink: TwoLink, p_goal: np.ndarray, obstacles: List[SphereObstacle] = None,
                   n_steps: int = 1000, eps: float = .1, m: float = 1.):

    obstacles = [] if obstacles is None else obstacles

    path = [twolink.end_position]
    potential = Potential(p_goal)
    step = 0
    while step <= n_steps:
        command = clfcbf_control(twolink, path[-1], potential, obstacles, m)
        path.append(path[-1] + eps*command)
        if np.linalg.norm(command) < 1e-3:
            break

        step += 1

    path = np.vstack(path)

    return path


if __name__ == '__main__':
    l1 = 1
    l2 = 1
    _twolink = TwoLink((l1, l2))

    goal = np.random.rand(2) * np.sqrt(l1**2 + l2**2)

    print(goal)

    path = clfcbf_planner(_twolink, goal)
    plt.plot(path[:, 0], path[:, 1], '-r')
    plt.show()