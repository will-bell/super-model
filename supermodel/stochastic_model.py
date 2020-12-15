import pathlib
from typing import List, Iterator

import torch
from torch import distributions
from torch import nn

from supermodel.fc_net import FCNet


class ModelEnsemble:

    n_models: int

    models: List[FCNet]

    def __init__(self, n_models: int, input_size: int, output_size: int, hidden_sizes: List[int] = None):
        self.n_models = n_models
        self.models = []
        for _ in range(n_models):
            self.models.append(FCNet(input_size, output_size, hidden_sizes))

    def __iter__(self) -> Iterator[FCNet]:
        return iter(self.models)

    def __call__(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = []
        for model in self.models:
            outputs.append(model(x))

        return outputs

    def sample(self, x: torch.Tensor, covariance: torch.Tensor = None) -> torch.Tensor:
        samples = []
        for model in self.models:
            gaussian = distributions.MultivariateNormal(model(x), covariance_matrix=covariance)
            samples.append(gaussian.rsample())
        samples = torch.vstack(samples)

        expectation = torch.mean(samples, dim=0)

        return expectation

    def zero_grad(self):
        for model in self.models:
            model.zero_grad()

    def to(self, device: torch.device):
        for i, model in enumerate(self.models):
            self.models[i] = model.to(device)

    def save(self, model_dir: pathlib.Path):
        model_dir.mkdir(exist_ok=True, parents=True)
        base = 'model-'
        for i, model in enumerate(self.models):
            path = model_dir / (base + f'{i}.pt')
            torch.save(model.state_dict(), str(path))

    @classmethod
    def load(cls, input_size: int, output_size: int, hidden_sizes: List[int], model_dir: pathlib.Path,
             device: torch.device = None) -> 'ModelEnsemble':
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model_paths = list(model_dir.glob('*pt'))
        model_ensemble = cls(len(model_paths), input_size, output_size, hidden_sizes)

        for i, model_path in enumerate(model_paths):
            model_ensemble.models[i].load_state_dict(torch.load(model_path, map_location=device))

        return model_ensemble


class GaussianModel(nn.Module):

    _net: FCNet

    covariance: torch.Tensor

    def __init__(self, net: FCNet, covariance: torch.Tensor = None):
        super().__init__()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._net = net.to(device)

        covariance = torch.eye(net.output_size) if covariance is None else covariance
        self.covariance = covariance.to(device)

    def __call__(self, x: torch.Tensor) -> distributions.MultivariateNormal:
        mean = self._net(x)

        return distributions.MultivariateNormal(mean, self.covariance)


if __name__ == '__main__':
    _model_ensemble = ModelEnsemble(3, 3, 2, [10, 10])

    for _model in _model_ensemble.models:
        print(_model)
