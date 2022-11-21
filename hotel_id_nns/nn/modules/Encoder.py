import copy
import torch
from typing import List
from torch import nn

from hotel_id_nns.nn.modules.ConvLayer import ConvLayer


class Encoder(nn.Module):
    def __init__(self, in_size: int, in_channels: int, hidden_channels: List[int]) -> None:
        super().__init__()

        channels = copy.deepcopy(hidden_channels)
        channels.insert(0, in_channels)

        self.out_size = torch.floor(in_size / 2) **  # TODO: continue

        modules = [
            ConvLayer(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                stride=2,
            ) for i in range(len(channels) - 1)
        ]

        self.encoder = nn.Sequential(*modules)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.encoder(input)
