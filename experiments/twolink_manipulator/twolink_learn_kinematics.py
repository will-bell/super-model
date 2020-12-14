from typing import Tuple, List

import numpy as np
import torch

from supermodel.learn_kinematics import backprop_clfcbf_control
from supermodel.learn_kinematics import learn_kinematics
from supermodel.potential import Potential
from supermodel.sphere_obstacle import SphereObstacle
from supermodel.stochastic_model import GaussianModel, FCNet
from experiments.twolink_manipulator.twolink import TwoLink


def torus_grid(delta: float):
    x_range = np.arange(-np.pi, np.pi, delta)
    y_range = np.arange(-np.pi, np.pi, delta)
    xx, yy = np.meshgrid(x_range, y_range)
    configurations = np.vstack([xx.ravel(), yy.ravel()]).T

    return configurations


def twolink_forward_kinematics(joint_angles: np.ndarray) -> np.ndarray:
    theta1, theta2 = joint_angles[:, 0], joint_angles[:, 1]
    l1, l2 = 1, 1

    # Calculate the trig values for more concise code
    s1 = np.sin(theta1)
    s12 = np.sin(theta1 + theta2)
    c1 = np.cos(theta1)
    c12 = np.cos(theta1 + theta2)

    joint_position = np.vstack([l1*c1, l1*s1]).T
    end_position = joint_position + np.vstack([l2 * c12, l2 * s12]).T

    return end_position


def twolink_backprop_planner(model: GaussianModel, c_start: np.ndarray, c_goal: np.ndarray,
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
    import pathlib

    l1 = 1
    l2 = 1

    theta_start = (np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi))
    twolink = TwoLink((l1, l2), joint_angles=theta_start)

    covariance = .01 * torch.eye(2)
    model = GaussianModel(FCNet(2, 2, [10, 10]), covariance=covariance)

    model_dir = pathlib.Path('./models')
    model_path = model_dir / 'twolink_test_model.pt'
    if not pathlib.Path.exists(model_path):
        torus_configurations = torus_grid(.001)
        trained_model, _ = learn_kinematics(model, twolink_forward_kinematics, torus_configurations, lr=1e-2,
                                            train_batch_size=500, valid_batch_size=100, n_epochs=5)

        model_dir.mkdir(exist_ok=True, parents=True)
        torch.save(trained_model.state_dict(), str(model_path))

    else:
        model.load_state_dict(torch.load(model_path))
        trained_model = model

    theta_goal = (np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi))
    p_goal = twolink.forward_kinematics2(np.array(theta_goal))

    _obstacles = []
    theta_path, _path_u = twolink_backprop_planner(trained_model, np.array(theta_start), np.array(p_goal),
                                                   obstacles=_obstacles, eps=.1, m=1000)

    fig1, ax1 = plt.subplots()
    ax1.plot(_path_u)

    fig, ax = plt.subplots()
    mag = 2

    def animate(i):
        ax.clear()
        ax.set_xlim([-mag, mag])
        ax.set_ylim([-mag, mag])
        ax.set_aspect('equal', 'box')
        twolink.update_joints(tuple(theta_path[i]))
        twolink.plot(ax)
        ax.plot(p_goal[:, 0], p_goal[:, 1], 'or')
        ax.plot([twolink.end_position[:, 0], p_goal[:, 0]], [twolink.end_position[:, 1], p_goal[:, 1]], '--g')


    anim = FuncAnimation(fig, animate, interval=5, frames=theta_path.shape[0] - 1)
    plt.draw()
    plt.show()
