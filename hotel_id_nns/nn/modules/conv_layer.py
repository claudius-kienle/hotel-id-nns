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
        bias: bool = True
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.non_linear = nn.ReLU()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        y = self.conv1(input)
        y = self.bn1(y)
        y = self.non_linear(y)
        return y