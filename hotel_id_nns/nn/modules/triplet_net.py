from turtle import forward
from torch import nn
from hotel_id_nns.nn.modules.resnet_chris import ResNet, resnet50_cfg
from hotel_id_nns.nn.modules.conv_layer import ConvLayer
from torch.nn import functional as F

class TripletNet(nn.Module):

    def __init__(self, lantent_dim: int = 128) -> None:
        super().__init__()

        self.backbone = ResNet(resnet50_cfg, out_features=256)


    def forward(x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = F.normalize(x, p=2, dim=1)
        return x

