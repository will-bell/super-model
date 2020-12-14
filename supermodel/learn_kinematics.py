# Ignore warnings
import warnings
from math import ceil
from typing import Callable, Union, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from supermodel.control import qp_min_effort
from supermodel.potential import Potential
from supermodel.sphere_obstacle import SphereObstacle, sphere_distance
from supermodel.stochastic_model import ModelEnsemble, GaussianModel, FCNet

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


def learn_kinematics(model_ensemble: ModelEnsemble, forward_kinematics: Kinematics, configurations: np.ndarray,
                     lr: float = 1e-2, n_epochs: int = 10, train_batch_size: int = 100, valid_batch_size: int = 10,
                     log_period: int = 20) -> Tuple[ModelEnsemble, np.ndarray]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_ensemble = model_ensemble.to(device)

    # Additional Info when using cuda
    if device.type == 'cuda':
        print(f'Using device: {device}')
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 2, 1), 'MB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 2, 1), 'MB')

    train_data, valid_data = shuffle_split(configurations)

    train_set = KinematicsDataset(train_data, forward_kinematics)
    train_generator = DataLoader(train_set, batch_size=train_batch_size, shuffle=True)

    valid_set = KinematicsDataset(valid_data, forward_kinematics)
    valid_generator = DataLoader(valid_set, batch_size=valid_batch_size, shuffle=True)

    optimizers = []
    for model in model_ensemble:
        optimizers.append(torch.optim.Adam(model.parameters(), lr=lr))

    loss_history = []

    n_total_updates = 0
    for epoch in range(n_epochs):
        n_epoch_updates = 0
        for configurations, true_states in train_generator:
            configurations = configurations.to(device)
            true_states = true_states.to(device)

            for model, optimizer in zip(model_ensemble, optimizers):
                optimizer.zero_grad()

                predicted_states = model(configurations)

                loss = F.mse_loss(predicted_states, true_states).mean()
                loss.backward()

                optimizer.step()

            if not n_epoch_updates % log_period:
                with torch.no_grad():
                    valid_count = 0
                    running_loss = 0
                    for valid_configs, valid_true_states in valid_generator:
                        valid_configs = valid_configs.to(device)
                        valid_true_states = valid_true_states.to(device)

                        batch_loss = 0
                        for model in model_ensemble.models:
                            predicted_states = model(valid_configs)
                            batch_loss += F.mse_loss(predicted_states, valid_true_states).mean().detach().to('cpu')

                        running_loss += batch_loss / model_ensemble.n_models

                        valid_count += 1

                    mean_loss = running_loss / valid_count
                    print(f'Epoch = {epoch + 1}, Total Updates = {n_total_updates}, Mean loss = {mean_loss}')
                    loss_history.append([n_total_updates, mean_loss])

            n_epoch_updates += 1
            n_total_updates += 1

    loss_history = np.asarray(loss_history)

    return model_ensemble, loss_history


def backprop_clfcbf_control(model_ensemble: ModelEnsemble, c_eval: np.ndarray, potential: Potential,
                            covariance: torch.Tensor, obstacles: List[SphereObstacle], m: float) -> np.ndarray:
    def zero_grads():
        model_ensemble.zero_grad()
        c_eval.grad.data.zero_()

    device = next(next(model_ensemble.__iter__()).parameters()).device

    c_eval = torch.Tensor(c_eval).to(device).requires_grad_()
    p_eval = model_ensemble.sample(c_eval, covariance)

    clf_value = potential.evaluate_potential(p_eval)
    clf_value.backward(retain_graph=True)

    Aclf = c_eval.grad.detach().to('cpu').clone().numpy()
    bclf = clf_value.detach().to('cpu').clone().item()

    zero_grads()

    if len(obstacles):
        Acbf = []
        bcbf = []
        for obstacle in obstacles:
            cbf_value = -sphere_distance(p_eval, obstacle)
            cbf_value.backward(retain_graph=True)

            Acbf.append(c_eval.grad.detach().to('cpu').clone().numpy())
            bcbf.append(cbf_value.detach().to('cpu').clone().item())

            zero_grads()

        Acbf = np.vstack(Acbf)
        bcbf = np.vstack(bcbf)

    else:
        Acbf = bcbf = None

    u_star = qp_min_effort(Aclf, bclf, Acbf, bcbf, m)

    zero_grads()

    return u_star


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    def identity_kinematics(blah: np.ndarray) -> np.ndarray:
        return blah @ np.eye(2)

    _model_ensemble = ModelEnsemble(100, 2, 2)

    x_range = np.arange(-10., 10., .1)
    y_range = np.arange(-10., 10., .1)
    xx, yy = np.meshgrid(x_range, y_range)
    _configurations = np.vstack([xx.ravel(), yy.ravel()]).T

    trained_model, _loss_history = learn_kinematics(_model_ensemble, identity_kinematics, _configurations, n_epochs=2, lr=1e-2)
    plt.plot(_loss_history[:, 0], _loss_history[:, 1], '-b')
    plt.xlabel('n updates')
    plt.ylabel('average ensemble loss')
    plt.show()
