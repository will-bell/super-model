import torch
from torch import nn
from typing import List


class FCNet(nn.Module):

    def __init__(self, input_size: int, output_size: int, hidden_sizes: List[int] = None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        hidden_sizes = [] if hidden_sizes is None else hidden_sizes
        if len(hidden_sizes) >= 1:
            input_layer = nn.Linear(input_size, hidden_sizes[0])
            if len(hidden_sizes) == 1:
                hidden_layers = []
            else:
                hidden_layers = [nn.Linear(s1, s2) for s1, s2 in zip(hidden_sizes[:-1], hidden_sizes[1:])]
            output_layer = nn.Linear(hidden_sizes[-1], output_size)
            linear_layers = [input_layer, *hidden_layers, output_layer]

        else:
            linear_layers = [nn.Linear(input_size, output_size)]

        network = [linear_layers[0]]
        for layer in linear_layers[1:]:
            network.append(nn.ReLU())
            network.append(layer)

        self.model = nn.Sequential(*network)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)

        return x


if __name__ == '__main__':
    net = FCNet(3, 3, [2])

    print(net(torch.Tensor([1., 2., 3.])))
