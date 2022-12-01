from torch import nn
import torch

class GlobalAvgPool2d(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert len(input.shape) >= 3

        output = torch.mean(input, dim=(-1, -2), keepdim=True)

        return output