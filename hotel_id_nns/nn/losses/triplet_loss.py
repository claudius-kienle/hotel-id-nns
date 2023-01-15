import torch
import torch.nn.functional as F

class TripletLoss():

    def __init__(self, alpha: float = 0.1) -> None:
        self.margin = alpha

    def __call__(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        inner = torch.sum((anchor - positive) ** 2, dim=-1)
        inter = torch.sum((anchor - negative) ** 2, dim=-1)

        losses = F.relu(inner - inter + self.margin)

        info = {
            "negative": inter.sum(),
            "positive": inner.sum()
        }

        return losses.sum(), info

