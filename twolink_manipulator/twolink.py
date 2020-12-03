from math import sin, cos
from typing import Tuple, Union, List

import matplotlib.pyplot as plt
import numpy as np

from twolink_manipulator.potential import Potential
from twolink_manipulator.sphere_obstacle import SphereObstacle
from twolink_manipulator.common import sphere_distance, sphere_distance_gradient

import cvxpy


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
        self.end_position = self.joint1_position = self.joint2_position = np.array([0., 0.])

        # Assign the missing parameters their values using forward kinematics
        self.update_joints(joint_angles)

    def update_joints(self, joint_angles: Tuple[float, float]):
        self.joint_angles = joint_angles
        self._forward_kinematics()

    def _forward_kinematics(self):
        theta0 = self.joint_angles[0]
        theta1 = self.joint_angles[1]

        l0 = self.link_lengths[0]
        l1 = self.link_lengths[1]

        self.joint2_position = self.joint1_position + np.array([l0*cos(theta0), l0*sin(theta0)])
        self.end_position = self.joint2_position + np.array([l1*cos(theta0 + theta1), l1*sin(theta0 + theta1)])

        # Calculate the trig values for more concise code
        s1 = sin(self.joint_angles[0])
        s12 = sin(self.joint_angles[0] + self.joint_angles[1])
        c1 = cos(self.joint_angles[0])
        c12 = cos(self.joint_angles[0] + self.joint_angles[1])

        # Unpack the link lengths
        l1, l2 = self.link_lengths

        # Construct the Jacobian matrix
        self.jacobian = np.array([[-l2 * s12 - l1 * s1, -l2 * s12], [l2 * c12 + l1 * c1, l2 * c12]])

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
            arm_length = self.link_lengths[0] + self.link_lengths[1]
            ax.set_xlim([-arm_length, arm_length])
            ax.set_ylim([-arm_length, arm_length])
            ax.set_aspect('equal', 'box')

        ax.plot([self.joint1_position[0], self.joint2_position[0]],
                [self.joint1_position[1], self.joint2_position[1]], 'r-')
        ax.plot([self.joint2_position[0], self.end_position[0]],
                [self.joint2_position[1], self.end_position[1]], 'r-')
        ax.plot(self.joint1_position[0], self.joint1_position[1], 'ko')
        ax.plot(self.joint2_position[0], self.joint2_position[1], 'ko')
        ax.plot(self.end_position[0], self.end_position[1], 'ko')
