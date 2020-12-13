# Ignore warnings
import warnings
from math import ceil
from typing import Callable, Union, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from common.stochastic_model import GaussianModel, FCNet

warnings.filterwarnings("ignore")

Kinematics = Callable[[np.ndarray], np.ndarray]


class KinematicsDataset(Dataset):

    def __init__(self, configurations: np.ndarray, forward_kinematics: Kinematics):
        self._configurations = torch.Tensor(configurations.copy()).float()
        self._states = torch.Tensor(forward_kinematics(configurations)).float()

    def __len__(self):
        return self._states.shape[0]

    def __getitem__(self, idx: Union[int, torch.Tensor]):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        configuration = self._configurations[idx, :]
        state = self._states[idx, :]

        return configuration, state


def shuffle_split(configurations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    configurations = configurations.copy()

    n = configurations.shape[0]
    train = ceil(4/5 * n)
    shuffle_indices = np.random.permutation(n)

    train_data = configurations[shuffle_indices[:train], :]
    valid_data = configurations[shuffle_indices[train:], :]

    return train_data, valid_data


def learn_kinematics(model: GaussianModel, forward_kinematics: Kinematics, configurations: np.ndarray, lr: float = 1e-2,
                     n_epochs: int = 10, train_batch_size: int = 100, valid_batch_size: int = 10,
                     log_period: int = 20) -> Tuple[GaussianModel, np.ndarray]:

    train_data, valid_data = shuffle_split(configurations)

    train_set = KinematicsDataset(train_data, forward_kinematics)
    train_generator = DataLoader(train_set, batch_size=train_batch_size, shuffle=True)

    valid_set = KinematicsDataset(valid_data, forward_kinematics)
    valid_generator = DataLoader(valid_set, batch_size=valid_batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []

    n_total_updates = 0
    for epoch in range(n_epochs):
        n_epoch_updates = 0
        for configurations, true_states in train_generator:
            optimizer.zero_grad()

            predicted_states = model(configurations).rsample()

            loss = F.mse_loss(predicted_states, true_states).mean()
            loss.backward()

            optimizer.step()
            n_epoch_updates += 1
            n_total_updates += 1

            if not n_epoch_updates % log_period:
                with torch.no_grad():
                    valid_count = 0
                    running_loss = 0
                    for valid_configs, valid_true_states in valid_generator:
                        predicted_states = model(valid_configs).rsample()
                        running_loss += F.mse_loss(predicted_states, valid_true_states).mean().detach()
                        valid_count += 1

                    mean_loss = running_loss / valid_count
                    print(f'Epoch = {epoch + 1}, Epoch Updates = {n_epoch_updates}, Mean loss = {mean_loss}')
                    loss_history.append([n_total_updates, mean_loss])

    loss_history = np.asarray(loss_history)

    return model, loss_history


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    def identity_kinematics(blah: np.ndarray) -> np.ndarray:
        return blah @ np.eye(2)

    covariance = .1 * torch.eye(2)
    _model = GaussianModel(FCNet(2, 2), covariance=covariance)

    x_range = np.arange(-10., 10., .1)
    y_range = np.arange(-10., 10., .1)
    xx, yy = np.meshgrid(x_range, y_range)
    _configurations = np.vstack([xx.ravel(), yy.ravel()]).T

    trained_model, _loss_history = learn_kinematics(_model, identity_kinematics, _configurations, n_epochs=10, lr=.8e-3)
    plt.plot(_loss_history[:, 0], _loss_history[:, 1], '-b')
    plt.xlabel('')
    plt.show()
