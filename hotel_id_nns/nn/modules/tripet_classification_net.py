from torch import nn
import torch
from typing import Mapping, Any
from hotel_id_nns.nn.modules.triplet_net import TripletNet
import torch


class TripletClassificationNet(nn.Module):

    def __init__(self, backbone: TripletNet, backbone_out_features: int, num_classes: int) -> None:
        super().__init__()

        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(backbone_out_features, 1024),
            nn.ReLU(True),
            # nn.Dropout(p=dropout),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            # nn.Dropout(p=dropout),
            nn.Linear(1024, num_classes),
        )
    
    def load_state_dict(self, state_dict: Mapping[str, Any]):
        if len(state_dict) == len(self.state_dict()):
            return super().load_state_dict(state_dict)
        else: # try to load weights into backbone
            self.backbone.load_state_dict(state_dict)

    def freeze_backbone(self):
        # only make head trainable
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.classifier(x)
        return x

