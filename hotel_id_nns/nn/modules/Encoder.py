import torch
from typing import List
from torch import nn

from hotel_id_nns.nn.modules.ConvLayer import ConvLayer


class Encoder(nn.Module):

    def __init__(self, in_channels: int, hidden_channels: List[int]) -> None:
        super().__init__()

        channels = hidden_channels.insert(0, in_channels)

        modules = [
            ConvLayer(in_channels=channels[i], out_channels=channels[i + 1])
            for i in range(len(channels))
        ]

        self.encoder = nn.Sequential(*modules)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.encoder(input)
