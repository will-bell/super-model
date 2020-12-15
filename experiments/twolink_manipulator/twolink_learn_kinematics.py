from typing import Tuple, List

import numpy as np
import torch

from supermodel.learn_kinematics import backprop_clfcbf_control
from supermodel.learn_kinematics import learn_kinematics
from supermodel.potential import Potential
from supermodel.sphere_obstacle import SphereObstacle
from supermodel.stochastic_model import ModelEnsemble
from experiments.twolink_manipulator.twolink import TwoLink


def torus_grid(delta: float) -> np.ndarray:
    x_range = np.arange(-np.pi, np.pi, delta)
    y_range = np.arange(-np.pi, np.pi, delta)
    xx, yy = np.meshgrid(x_range, y_range)
    configurations = np.vstack([xx.ravel(), yy.ravel()]).T
    configurations = (configurations + np.random.uniform(-delta / 4., delta / 4., configurations.shape)) % (2 * np.pi)

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


def twolink_backprop_planner(twolink: TwoLink, model_ensemble: ModelEnsemble, c_start: np.ndarray, c_goal: np.ndarray,
                             covariance: torch.Tensor = None, obstacles: List[SphereObstacle] = None,
                             n_steps: int = 1000, eps: float = .1, m: float = 100.) -> Tuple[np.ndarray, np.ndarray]:
    """

    Args:
        model_ensemble:
        c_start:
        c_goal:
        covariance:
        obstacles:
        n_steps:
        eps:
        m:

    Returns:

    """
    device = next(next(model_ensemble.__iter__()).parameters()).device

    covariance = torch.eye(c_start.size).to(device) if covariance is None else covariance
    obstacles = [] if obstacles is None else obstacles

    path = [c_start % (2 * np.pi)]
    potential = Potential(torch.Tensor(c_goal), quadratic_radius=.01)

    p_start, _ = twolink.forward_kinematics((c_start[0], c_start[1]))
    path_u = [potential.evaluate_potential(p_start)]

    step = 0
    while step <= n_steps:
        command = backprop_clfcbf_control(model_ensemble, path[-1], potential, covariance, obstacles, m).squeeze()

        c_next = (path[-1] + eps * command) % (2 * np.pi)
        p_next, _ = twolink.forward_kinematics((c_next[0], c_next[1]))

        path.append(c_next)
        path_u.append(potential.evaluate_potential(p_next))

        if np.linalg.norm(command) < 1e-2 or np.linalg.norm(path[-1] - c_goal) < 1e-2:
            break

        step += 1

    path = np.vstack(path)
    path_u = np.array(path_u)

    return path, path_u


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from experiments.twolink_manipulator.twolink_clfcbf_planner import sample_good_theta
    import pathlib

    model_dir = pathlib.Path('./models')
    model_path = model_dir / 'twolink_test'
    if not pathlib.Path.exists(model_path):
        torus_configurations = torus_grid(.005)
        _model_ensemble = ModelEnsemble(100, 2, 2, [512, 512])
        _model_ensemble, _ = learn_kinematics(_model_ensemble, twolink_forward_kinematics, torus_configurations,
                                              lr=1e-3, train_batch_size=500, valid_batch_size=100, n_epochs=3)

        _model_ensemble.save(model_path)

    else:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _model_ensemble = ModelEnsemble.load(2, 2, [512, 512], model_path, _device)

    l1 = 1
    l2 = 1

    _twolink = TwoLink((l1, l2), joint_angles=(0., 0.))

    _obstacles = [SphereObstacle((1, -.5), .25)]
    theta_start = sample_good_theta(_twolink, _obstacles)
    _theta_goal = sample_good_theta(_twolink, _obstacles)
    p_goal, _ = _twolink.forward_kinematics(_theta_goal)

    _twolink.update_joints(theta_start)

    _covariance = .1 * torch.eye(2)
    theta_path, _path_u = twolink_backprop_planner(_twolink, _model_ensemble, np.array(theta_start), np.array(p_goal),
                                                   covariance=_covariance, obstacles=_obstacles, eps=.1, m=1000.)

    fig1, ax1 = plt.subplots()
    ax1.plot(_path_u)
    ax1.xlabel('steps')
    ax1.ylabel('potential')

    fig2, ax2 = plt.subplots()
    ax2.plot(theta_path[:, 0], '-b', label=r'$\theta_1$')
    ax2.plot(theta_path[:, 1], '-r', label=r'$\theta_2$')
    ax2.set_xlabel('steps')
    ax2.set_ylabel(r'$\theta$ [radians]')
    ax2.legend()

    fig, ax = plt.subplots()
    mag = 2

    def animate(i):
        ax.clear()
        ax.xlabel('X')
        ax.ylabel('Y')
        ax.set_xlim([-mag, mag])
        ax.set_ylim([-mag, mag])
        ax.set_aspect('equal', 'box')
        _twolink.update_joints(tuple(theta_path[i]))
        _twolink.plot(ax)
        ax.plot(p_goal[:, 0], p_goal[:, 1], 'or')
        ax.plot([_twolink.end_position[:, 0], p_goal[:, 0]], [_twolink.end_position[:, 1], p_goal[:, 1]], '--g')
        for obstacle in _obstacles:
            obstacle.plot(ax)

    anim = FuncAnimation(fig, animate, interval=5, frames=theta_path.shape[0] - 1)
    plt.draw()
    plt.show()
