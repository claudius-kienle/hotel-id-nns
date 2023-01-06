import torch
from torch import nn


class ConvLayer(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()

        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        batch_norm = nn.BatchNorm2d(num_features=out_channels)
        non_linear = nn.ReLU()

        self.layer = nn.Sequential(
            conv,
            batch_norm,
            non_linear,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layer(input)