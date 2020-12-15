import numpy as np
import torch

from supermodel.learn_kinematics import learn_kinematics
from supermodel.sphere_obstacle import SphereObstacle
from supermodel.stochastic_model import ModelEnsemble
from experiments.twolink_manipulator.twolink import TwoLink
from experiments.twolink_manipulator.twolink_learn_kinematics import torus_grid, twolink_forward_kinematics, \
    twolink_backprop_planner

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import pathlib

    model_dir = pathlib.Path('./models')
    model_path = model_dir / 'twolink_test'
    if not pathlib.Path.exists(model_path):
        torus_configurations = torus_grid(.005)
        model_ensemble = ModelEnsemble(100, 2, 2, [512, 512])
        model_ensemble, _ = learn_kinematics(model_ensemble, twolink_forward_kinematics, torus_configurations,
                                              lr=1e-3, train_batch_size=500, valid_batch_size=100, n_epochs=3)

        model_ensemble.save(model_path)

    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_ensemble = ModelEnsemble.load(2, 2, [512, 512], model_path, device)

    l1 = 1
    l2 = 1

    twolink = TwoLink((l1, l2), joint_angles=(0., 0.))

    obstacles = [SphereObstacle((.75, -.75), .5), SphereObstacle((-.75, .75), .5)]

    starts = [(-1.25, 1.25), (.25, -.25), (.75, 0.)]
    goals = [(.25, -.25), (1.25, -1.2), (1.25, -1.2)]
    for start, goal in zip(starts, goals):
        theta_start = twolink.inverse_kinematics(start)
        theta_goal = twolink.inverse_kinematics(goal)
        p_goal, _ = twolink.forward_kinematics(theta_goal)

        twolink.update_joints(theta_start)

        _covariance = .1 * torch.eye(2)
        theta_path, _path_u = twolink_backprop_planner(twolink, model_ensemble, np.array(theta_start), np.array(p_goal),
                                                       covariance=_covariance, obstacles=obstacles, eps=.01, m=1000.)

        x_path = []
        for theta in theta_path:
            x, _ = twolink.forward_kinematics((theta[0], theta[1]))
            x_path.append(x)
        x_path = np.vstack(x_path)

        fig0, ax0 = plt.subplots()
        ax0.plot(x_path[:, 0], x_path[:, 1], '--b')
        for obstacle in obstacles:
            obstacle.plot(ax0)
        twolink.update_joints((theta_path[0, 0], theta_path[0, 1]))
        twolink.plot(ax0)
        twolink.update_joints((theta_path[-1, 0], theta_path[-1, 1]))
        twolink.plot(ax0)
        ax0.plot(p_goal[:, 0], p_goal[:, 1], 'og')
        ax0.set_aspect('equal', 'box')
        ax0.set_xlabel('X')
        ax0.set_ylabel('Y')

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

        fig, ax3 = plt.subplots()
        mag = 2

        def animate(i):
            ax3.clear()
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.set_xlim([-mag, mag])
            ax3.set_ylim([-mag, mag])
            ax3.set_aspect('equal', 'box')
            twolink.update_joints(tuple(theta_path[i]))
            twolink.plot(ax3)
            ax3.plot(p_goal[:, 0], p_goal[:, 1], 'or')
            ax3.plot([twolink.end_position[:, 0], p_goal[:, 0]], [twolink.end_position[:, 1], p_goal[:, 1]], '--g')
            for obstacle in obstacles:
                obstacle.plot(ax3)

        anim = FuncAnimation(fig, animate, interval=5, frames=theta_path.shape[0] - 1)
        plt.draw()
        plt.show()