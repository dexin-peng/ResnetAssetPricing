import torch.nn as nn


class NN(nn.Module):
    def __init__(self, in_features: int, hidden_sizes: list[int]):
        super().__init__()
        layers = []
        last = in_features
        for h in hidden_sizes:
            layers += [nn.Linear(last, h)]
            if h != last:
                layers += [nn.BatchNorm1d(h)]
            layers += [nn.ReLU()]
            last = h
        self.output_layer = nn.Linear(last, 1)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.output_layer(self.net(x))
