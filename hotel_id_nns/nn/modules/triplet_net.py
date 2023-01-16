from turtle import forward
from torch import nn
import torch
from hotel_id_nns.nn.modules.resnet_chris import ResNet, resnet50_cfg
from hotel_id_nns.nn.modules.conv_layer import ConvLayer
import torch
from torch.nn import functional as F

class TripletNet(nn.Module):

    def __init__(self, backbone: torch.nn.Module) -> None:
        super().__init__()

        self.backbone = backbone


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = F.normalize(x, p=2, dim=1)
        return x

