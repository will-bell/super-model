from typing import List, Tuple

import numpy as np
import torch

from common.learn_kinematics import learn_kinematics, backprop_clfcbf_control
from common.potential import Potential
from common.sphere_obstacle import SphereObstacle
from common.stochastic_model import GaussianModel, FCNet


def point_mass_backprop_planner(model: GaussianModel, c_start: np.ndarray, c_goal: np.ndarray,
                                obstacles: List[SphereObstacle] = None, n_steps: int = 1000, eps: float = .1,
                                m: float = 100.) -> Tuple[np.ndarray, np.ndarray]:

    obstacles = [] if obstacles is None else obstacles

    path = [c_start]
    potential = Potential(torch.Tensor(c_goal), quadratic_radius=.05)
    path_u = [potential.evaluate_potential(torch.Tensor(c_start)).item()]

    step = 0
    while step <= n_steps:
        command = backprop_clfcbf_control(model, path[-1], potential, obstacles, m).squeeze()
        path.append(path[-1] + eps * command)

        path_u.append(potential.evaluate_potential(torch.Tensor(path[-1])))

        if np.linalg.norm(command) < 1e-2 or np.linalg.norm(path[-1] - c_goal) < 1e-2:
            break

        step += 1

    path = np.vstack(path)
    path_u = np.array(path_u)

    return path, path_u


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation


    def identity_kinematics(blah: np.ndarray) -> np.ndarray:
        return blah @ np.eye(2)


    covariance = .01 * torch.eye(2)
    _model = GaussianModel(FCNet(2, 2), covariance=covariance)

    x_range = np.arange(-10., 10., .1)
    y_range = np.arange(-10., 10., .1)
    xx, yy = np.meshgrid(x_range, y_range)
    _configurations = np.vstack([xx.ravel(), yy.ravel()]).T

    trained_model, _ = learn_kinematics(_model, identity_kinematics, _configurations, n_epochs=1, lr=1e-2)

    _c_start = np.array([-1.5, 0.1])
    _c_goal = np.array([1.5, 0.])

    _obstacles = [SphereObstacle((0., 0.), .25), SphereObstacle((0., 1.), .25)]
    _path, _path_u = point_mass_backprop_planner(trained_model, _c_start, _c_goal, obstacles=_obstacles, eps=.1, m=1000)

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
        ax.plot(_c_goal[0], _c_goal[1], 'or')
        for obstacle in _obstacles:
            obstacle.plot(ax)
        ax.plot([_path[i, 0], _c_goal[0]], [_path[i, 1], _c_goal[1]], '--g')


    anim = FuncAnimation(fig, animate, interval=5, frames=_path.shape[0] - 1)
    plt.draw()
    plt.show()
