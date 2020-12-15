from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from supermodel.sphere_obstacle import sphere_distance, sphere_distance_gradient
from supermodel.control import qp_min_effort
from supermodel.potential import Potential
from supermodel.sphere_obstacle import SphereObstacle
from experiments.twolink_manipulator.twolink import TwoLink


def twolink_clfcbf_control(twolink: TwoLink, theta_eval: np.ndarray, potential: Potential,
                           obstacles: List[SphereObstacle], m: float) -> np.ndarray:

    p_eval, jacobian = twolink.forward_kinematics(tuple(theta_eval))
    Aclf = (jacobian.T @ potential.evaluate_potential_gradient(p_eval).reshape(2, 1)).reshape(1, 2)
    bclf = potential.evaluate_potential(p_eval)

    if len(obstacles):
        Acbf = []
        bcbf = []
        for obstacle in obstacles:
            Acbf.append((jacobian.T @ -sphere_distance_gradient(p_eval, obstacle).reshape(2, 1)).reshape(1, 2))
            bcbf.append(-sphere_distance(p_eval, obstacle))
        Acbf = np.vstack(Acbf)
        bcbf = np.vstack(bcbf)

    else:
        Acbf = bcbf = None

    u_star = qp_min_effort(Aclf, bclf, Acbf, bcbf, m)

    return u_star


def twolink_clfcbf_planner(twolink: TwoLink, theta_goal: Tuple[float, float], obstacles: List[SphereObstacle] = None,
                           n_steps: int = 1000, eps: float = 1., m: float = 10000.) -> Tuple[np.ndarray, np.ndarray]:

    obstacles = [] if obstacles is None else obstacles

    path = [np.array(twolink.joint_angles) % (2 * np.pi)]
    p_start = twolink.end_position
    p_goal, _ = twolink.forward_kinematics(theta_goal)
    potential = Potential(p_goal, quadratic_radius=.05)
    path_u = [potential.evaluate_potential(p_start)]

    step = 0
    while step <= n_steps:
        command = twolink_clfcbf_control(twolink, path[-1], potential, obstacles, m).squeeze()

        c_next = (path[-1] + eps*command) % (2 * np.pi)
        p_next, _ = twolink.forward_kinematics(tuple(path[-1]))

        path.append(c_next)
        path_u.append(potential.evaluate_potential(p_next))

        if np.linalg.norm(command) < 1e-2 or np.linalg.norm(path[-1] - p_goal) < 1e-2:
            break

        step += 1

    path = np.vstack(path)
    path_u = np.array(path_u)

    return path, path_u


def sample_good_theta(twolink: TwoLink, obstacles: List[SphereObstacle]) -> Tuple[float, float]:
    def test_point():
        for obstacle in obstacles:
            if sphere_distance(point, obstacle) < 0:
                return False

        return True

    while True:
        theta = (np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi))
        point, _ = twolink.forward_kinematics(theta)

        if test_point():
            break

    return theta


if __name__ == '__main__':
    from math import sqrt

    l1 = 1
    l2 = 1

    _twolink = TwoLink((l1, l2), joint_angles=(0., 0.))

    _obstacles = [SphereObstacle((1, -.5), .25)]
    theta_start = sample_good_theta(_twolink, _obstacles)
    _theta_goal = sample_good_theta(_twolink, _obstacles)
    _p_goal, _ = _twolink.forward_kinematics(_theta_goal)

    _twolink.update_joints(theta_start)

    theta_path, _path_u = twolink_clfcbf_planner(_twolink, _theta_goal, eps=.1, m=1000., obstacles=_obstacles)

    fig1, ax1 = plt.subplots()
    ax1.plot(_path_u)
    ax1.set_xlabel('steps')
    ax1.set_ylabel('potential')

    fig2, ax2 = plt.subplots()
    ax2.plot(theta_path[:, 0], '-b', label=r'$\theta_1$')
    ax2.plot(theta_path[:, 1], '-r', label=r'$\theta_2$')
    ax2.set_xlabel('steps')
    ax2.set_ylabel(r'$\theta$ [radians]')
    ax2.legend()

    fig, ax = plt.subplots()
    mag = l1 + l2

    def animate(i):
        ax.clear()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_xlim([-mag, mag])
        ax.set_ylim([-mag, mag])
        ax.set_aspect('equal', 'box')
        _twolink.update_joints(tuple(theta_path[i]))
        _twolink.plot(ax)
        ax.plot(_p_goal[:, 0], _p_goal[:, 1], 'or')
        ax.plot([_twolink.end_position[:, 0], _p_goal[:, 0]], [_twolink.end_position[:, 1], _p_goal[:, 1]], '--g')
        for obstacle in _obstacles:
            obstacle.plot(ax)

    anim = FuncAnimation(fig, animate, interval=5, frames=theta_path.shape[0]-1)
    plt.draw()
    plt.show()
