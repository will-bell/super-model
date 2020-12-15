from math import sin, cos, acos, atan2
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


class TwoLink:

    link_lengths: Union[Tuple[float, float], None]

    joint_angles: Union[Tuple[float, float], None]

    joint1_position: Union[np.ndarray, None]

    joint2_position: Union[np.ndarray, None]

    end_position: Union[np.ndarray, None]

    jacobian: Union[np.ndarray, None]

    def __init__(self, link_lengths: Tuple[float, float], joint_angles: Tuple[float, float] = (0., 0.)):
        self.link_lengths = link_lengths

        # Initialize these attributes here to be clear (all attributes should be defined in __init__)
        self.joint_angles = self.jacobian = None
        self.end_position = self.joint1_position = self.joint2_position = np.array([[0., 0.]])

        # Assign the missing parameters their values using forward kinematics
        self.update_joints(joint_angles)

    def update_joints(self, joint_angles: Tuple[float, float]):
        self.joint_angles = joint_angles
        self.joint2_position, self.end_position, self.jacobian = self._forward_kinematics(np.array(joint_angles))

    def _forward_kinematics(self, joint_angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(joint_angles.shape) < 2:
            joint_angles = joint_angles[np.newaxis]
        theta1, theta2 = joint_angles[:, 0], joint_angles[:, 1]
        l1, l2 = self.link_lengths

        # Calculate the trig values for more concise code
        s1 = np.sin(theta1)[np.newaxis].T
        s12 = np.sin(theta1 + theta2)[np.newaxis].T
        c1 = np.cos(theta1)[np.newaxis].T
        c12 = np.cos(theta1 + theta2)[np.newaxis].T

        joint2_position = self.joint1_position + np.hstack([l1*c1, l1*s1])
        end_position = joint2_position + np.hstack([l2*c12, l2*s12])

        # Construct the Jacobian matrix
        jacobian = np.array([[-l2 * s12 - l1 * s1, -l2 * s12], [l2 * c12 + l1 * c1, l2 * c12]]).squeeze()

        return joint2_position, end_position, jacobian

    def forward_kinematics(self, joint_angles: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        _, end_position, jacobian = self._forward_kinematics(np.array(joint_angles))

        return end_position, jacobian

    def forward_kinematics2(self, joint_angles: np.ndarray) -> np.ndarray:
        _, end_position, _ = self._forward_kinematics(joint_angles)

        return end_position

    def inverse_kinematics(self, end_position: Tuple[float, float]) -> Tuple[float, float]:
        x, y = end_position
        l1, l2 = self.link_lengths
        theta2 = acos((x**2 + y**2 - l1**2 - l2**2) / (2*l1*l2))
        theta1 = atan2(y, x) - atan2(l2*sin(theta2), l1 + l1*cos(theta2))

        return theta1, theta2

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
            arm_length = self.link_lengths[0] + self.link_lengths[1]
            ax.set_xlim([-arm_length, arm_length])
            ax.set_ylim([-arm_length, arm_length])
            ax.set_aspect('equal', 'box')

        ax.plot([self.joint1_position[:, 0], self.joint2_position[:, 0]],
                [self.joint1_position[:, 1], self.joint2_position[:, 1]], 'b-')
        ax.plot([self.joint2_position[:, 0], self.end_position[:, 0]],
                [self.joint2_position[:, 1], self.end_position[:, 1]], 'b-')
        ax.plot(self.joint1_position[:, 0], self.joint1_position[:, 1], 'ko')
        ax.plot(self.joint2_position[:, 0], self.joint2_position[:, 1], 'ko')
        ax.plot(self.end_position[:, 0], self.end_position[:, 1], 'ko')
