import torch
from torch import nn
from torch import distributions

from common.fc_net import FCNet


class GaussianModel(nn.Module):

    def __init__(self, net: FCNet, covariance: torch.Tensor = None):
        super().__init__()
        self._net = net
        self.covariance = torch.eye(net.output_size) if covariance is None else covariance

    def __call__(self, x: torch.Tensor) -> distributions.MultivariateNormal:
        mean = self._net(x)

        return distributions.MultivariateNormal(mean, self.covariance)


if __name__ == '__main__':
    fcnet = FCNet(2, 2)
    model = GaussianModel(fcnet)

    print(list(model.parameters()))
