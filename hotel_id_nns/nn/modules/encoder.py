import copy
from math import floor
import torch
from typing import List
from torch import nn

from hotel_id_nns.nn.modules.conv_layer import ConvLayer


class Encoder(nn.Module):
    def __init__(self, in_size: int, in_channels: int, hidden_channels: List[int],
        stride:int = 2,
        padding:int = 1,
        kernel_size:int = 3,
    ) -> None:
        super().__init__()

        channels = copy.deepcopy(hidden_channels)
        channels.insert(0, in_channels)

        out_size = in_size
        for _ in range(len(hidden_channels)):
            out_size = floor((out_size - kernel_size + 2 * padding) / stride + 1)
        self.__out_size = out_size

        modules = [
            ConvLayer(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                padding=padding,
                stride=stride,
                kernel_size=kernel_size
            ) for i in range(len(channels) - 1)
        ]

        self.encoder = nn.Sequential(*modules)

    @property
    def out_size(self) -> int:
        return self.__out_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.encoder(input)
