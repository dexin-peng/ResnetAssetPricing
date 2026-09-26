import torch
import torch.nn as nn

class FeedForwardBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_features, out_features)]
        self.norm = nn.BatchNorm1d(out_features)
        self.act = nn.ReLU()

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.net(x)))


class ResidualBlock(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        layers: list[nn.Module] = []
        layers.append(nn.Linear(d, d))
        self.net = nn.Sequential(*layers)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.net(x) + x)



class ResNet(nn.Module):
    def __init__(self, in_features: int, hidden_sizes: list[int] | tuple[int, ...]):
        super().__init__()

        blocks: list[nn.Module] = []
        prev = int(in_features)

        for i, cur in enumerate(hidden_sizes):
            cur = int(cur)
            if prev == cur:
                blocks.append(ResidualBlock(d=cur))
            else:
                blocks.append(FeedForwardBlock(prev, cur))
            prev = cur

        self.net = nn.Sequential(*blocks)
        self.output_layer = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.net(x))
