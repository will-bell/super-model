from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from common.sphere_obstacle import sphere_distance, sphere_distance_gradient
from common.control import qp_min_effort
from common.potential import Potential
from common.sphere_obstacle import SphereObstacle


def point_mass_clfcbf_control(p_eval: np.ndarray, potential: Potential, obstacles: List[SphereObstacle], m: float) -> np.ndarray:
    Aclf = potential.evaluate_potential_gradient(p_eval)
    bclf = potential.evaluate_potential(p_eval)

    if len(obstacles):
        Acbf = []
        bcbf = []
        for obstacle in obstacles:
            Acbf.append(-sphere_distance_gradient(p_eval, obstacle))
            bcbf.append(-sphere_distance(p_eval, obstacle))
        Acbf = np.vstack(Acbf)
        bcbf = np.vstack(bcbf)

    else:
        Acbf = bcbf = None

    u_star = qp_min_effort(Aclf, bclf, Acbf, bcbf, m)

    return u_star


def point_mass_clfcbf_planner(p_start: np.ndarray, p_goal: np.ndarray, obstacles: List[SphereObstacle] = None,
                              n_steps: int = 1000, eps: float = .1, m: float = 100.) -> Tuple[np.ndarray, np.ndarray]:

    obstacles = [] if obstacles is None else obstacles

    path = [p_start]
    potential = Potential(p_goal, quadratic_radius=.05)
    path_u = [potential.evaluate_potential(p_start)]

    step = 0
    while step <= n_steps:
        command = point_mass_clfcbf_control(path[-1], potential, obstacles, m).squeeze()
        path.append(path[-1] + eps * command)

        path_u.append(potential.evaluate_potential(path[-1]))

        if np.linalg.norm(command) < 1e-2 or np.linalg.norm(path[-1] - p_goal) < 1e-2:
            break

        step += 1

    path = np.vstack(path)
    path_u = np.array(path_u)

    return path, path_u


if __name__ == '__main__':
    _p_start = np.array([-1.5, 0.1])
    _p_goal = np.array([1.5, 0.])

    _obstacles = [SphereObstacle((0., 0.), .25)]
    _path, _path_u = point_mass_clfcbf_planner(_p_start, _p_goal, obstacles=_obstacles, eps=.1, m=100)

    fig1, ax1 = plt.subplots()
    ax1.plot(_path_u)

    fig, ax = plt.subplots()
    mag = 2


    def animate(i):
        ax.clear()
        ax.set_xlim([-mag, mag])
        ax.set_ylim([-mag, mag])
        ax.set_aspect('equal', 'box')
        ax.plot(_path[i, 0], _path[i, 1], 'ob')
        ax.plot(_path[:i, 0], _path[:i, 1], '-b')
        ax.plot(_p_goal[0], _p_goal[1], 'or')
        for obstacle in _obstacles:
            obstacle.plot(ax)
        ax.plot([_path[i, 0], _p_goal[0]], [_path[i, 1], _p_goal[1]], '--g')


    anim = FuncAnimation(fig, animate, interval=5, frames=_path.shape[0] - 1)
    plt.draw()
    plt.show()
