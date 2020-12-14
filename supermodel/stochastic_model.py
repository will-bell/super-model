import torch
from torch import nn
from torch import distributions

from supermodel.fc_net import FCNet


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
    fcnet = FCNet(2, 2)
    model = GaussianModel(fcnet)

    print(list(model.parameters()))
