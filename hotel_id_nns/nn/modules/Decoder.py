import copy
import torch
from typing import List
from torch import nn

from hotel_id_nns.nn.modules.ConvLayer import ConvLayer


class Decoder(nn.Module):

    def __init__(self, out_channels: int, hidden_channels: List[int]) -> None:
        super().__init__()

        channels = copy.deepcopy(hidden_channels)
        channels.append(out_channels)

        modules = [
            ConvLayer(in_channels=channels[i], out_channels=channels[i + 1])
            for i in range(len(channels) - 1)
        ]

        self.decoder = nn.Sequential(*modules)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.decoder(input)
