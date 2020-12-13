from common.learn_kinematics import learn_kinematics
from common.stochastic_model import GaussianModel, FCNet
import torch
import numpy as np


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


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    covariance = .1 * torch.eye(2)
    model = GaussianModel(FCNet(2, 2, [10, 10]), covariance=covariance)
    torus_configurations = torus_grid(.01)

    trained_model, loss_history = learn_kinematics(model, twolink_forward_kinematics, torus_configurations,
                                                   lr=1e-2, train_batch_size=800, valid_batch_size=100, n_epochs=5)
    plt.plot(loss_history[:, 0], loss_history[:, 1], '-b')
    plt.xlabel('')
    plt.show()
