import copy
from math import floor
from typing import List, Tuple
import torch
from torch import nn
from hotel_id_nns.nn.modules.conv_layer import ConvLayer
from hotel_id_nns.nn.modules.decoder import Decoder
from hotel_id_nns.nn.modules.global_avg_pool_2d import GlobalAvgPool2d


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

        modules = [
            ConvLayer(in_channels=channels[i],
                      out_channels=channels[i + 1],
                      padding=padding,
                      stride=stride,
                      kernel_size=kernel_size) for i in range(len(channels) - 1)
        ]

        self.global_avg_pool = GlobalAvgPool2d()

        fcs = [
            nn.Linear(in_features=channels[-1], out_features=channels[-1]),
            nn.Linear(in_features=channels[-1], out_features=num_classes)
        ]

        self.backbone = nn.Sequential(*modules)
        self.fcs = nn.Sequential(
            nn.Flatten(),
            *fcs,
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        features = self.backbone(input)
        features = self.global_avg_pool(features)
        class_probs = self.fcs(features)
        return class_probs