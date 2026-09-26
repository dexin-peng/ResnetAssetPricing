import torch
import torch.nn as nn
import torch.nn.functional as F
from model.nnp import FeedForwardBlock


class TypeIResidualBlock(nn.Module):
    def __init__(
        self,
        sizes: int,
        dropout: float = 0.0,
        negative_slope: float = 0.01,
    ):
        super().__init__()

        self.skip = nn.Identity()

        self.norm = nn.LayerNorm(sizes)
        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=False)
        self.fc = nn.Linear(sizes, sizes, bias=False)

        self.use_dropout = dropout > 0.0
        if self.use_dropout:
            self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = self.fc(self.act(self.norm(x)))

        if self.use_dropout:
            out = self.drop(out)
        return self.act(self.skip(x) + out)


class TypeIIResidualBlock(nn.Module):
    def __init__(
        self,
        in_sizes: int,
        out_sizes: int,
        dropout: float = 0.0,
        negative_slope: float = 0.01,
    ):
        super().__init__()
        self.in_sizes = in_sizes
        self.out_sizes = out_sizes

        self.norm = nn.LayerNorm(in_sizes)
        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=False)
        self.fc = nn.Linear(in_sizes, out_sizes, bias=False)

        self.use_dropout = dropout > 0.0
        if self.use_dropout:
            self.drop = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if isinstance(inputs, tuple):
            x, accumulator = inputs
        else:
            x, accumulator = inputs, None

        out = self.fc(self.act(self.norm(x)))
        if self.use_dropout:
            out = self.drop(out)
        if accumulator is not None:
            out = out + accumulator
        return self.act(out)


class NonLinearProjectionBlock(nn.Module):
    def __init__(
        self,
        in_sizes: int,
        out_sizes: int,
        negative_slope: float = 0.01,
    ):
        super().__init__()
        self.in_sizes = in_sizes
        self.out_sizes = out_sizes

        self.skip = nn.Linear(in_sizes, out_sizes, bias=False)
        self.norm = nn.LayerNorm(out_sizes)
        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=False)

    def forward(self, inputs: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(inputs, tuple):
            x, accumulator = inputs
        else:
            x, accumulator = inputs, None

        proj = self.act(self.norm(self.skip(x)))
        accumulator = proj if accumulator is None else (accumulator + proj)
        return x, accumulator


class ResNetp(nn.Module):
    def __init__(
        self,
        hidden_sizes: list[int],
        dropout: float = 0.0,
        negative_slope: float = 0.01,
    ):
        super().__init__()

        blocks = []
        i = 0
        n = len(hidden_sizes)
        while i < n:
            in_sz, out_sz = hidden_sizes[i]

            if in_sz == out_sz:
                blocks.append(
                    TypeIResidualBlock(
                        in_sz,
                        dropout=dropout,
                        negative_slope=negative_slope,
                    )
                )
                i += 1
                continue

            run_len = 1
            while (i + run_len < n) and (hidden_sizes[i + run_len] == [in_sz, out_sz]):
                run_len += 1

            if run_len == 1:
                blocks.append(
                    FeedForwardBlock(
                        in_sizes=in_sz,
                        out_sizes=out_sz,
                        dropout=dropout,
                        negative_slope=negative_slope,
                    )
                )
            else:
                for _ in range(run_len - 1):
                    blocks.append(
                        NonLinearProjectionBlock(
                            in_sizes=in_sz,
                            out_sizes=out_sz,
                            negative_slope=negative_slope,
                        )
                    )
                blocks.append(
                    TypeIIResidualBlock(
                        in_sizes=in_sz,
                        out_sizes=out_sz,
                        dropout=dropout,
                        negative_slope=negative_slope,
                    )
                )
            i += run_len

        self.last = hidden_sizes[-1][1]

        self.blocks = nn.Sequential(*blocks)
        self.norm_out = nn.LayerNorm(self.last)
        self.output_layer = nn.Linear(self.last, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=negative_slope, mode="fan_in")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        x = self.norm_out(x)
        return self.output_layer(x)
