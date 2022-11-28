import copy
from math import floor
from typing import List, Tuple
import torch
from torch import nn
from hotel_id_nns.nn.modules.ConvLayer import ConvLayer
from hotel_id_nns.nn.modules.Decoder import Decoder


class ClassNet(nn.Module):
    def __init__(
        self,
        in_size: int,
        in_channels: int,
        hidden_channels: List[int],
        num_classes: int,
        name: str = 'ClassNet',
    ) -> None:
        super().__init__()

        self.name = name

        channels = copy.deepcopy(hidden_channels)
        channels.insert(0, in_channels)

        padding = 1
        kernel_size = 3
        stride = 2

        out_size = in_size
        for _ in range(len(hidden_channels)):
            out_size = floor((out_size - kernel_size + 2 * padding) / stride + 1)

        latent_size = out_size**2 * channels[-1]

        modules = [
            ConvLayer(in_channels=channels[i],
                      out_channels=channels[i + 1],
                      padding=padding,
                      stride=stride,
                      kernel_size=kernel_size) for i in range(len(channels) - 1)
        ]

        fcs = [
            nn.Linear(in_features=latent_size, out_features=latent_size),
            nn.Linear(in_features=latent_size, out_features=num_classes)
        ]

        self.backbone = nn.Sequential(*modules)
        self.fcs = nn.Sequential(
            nn.Flatten(),
            *fcs,
            nn.Softmax(dim=-1),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        features = self.backbone(input)
        class_probs = self.fcs(features)
        return class_probs