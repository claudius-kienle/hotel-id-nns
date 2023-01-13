import torch

class TripletLoss():

    def __init__(self, alpha: float = 0.1) -> None:
        self.l2loss = torch.nn.MSELoss(reduction='none')
        self.alpha = alpha

    def __call__(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        inner = self.l2loss(anchor, positive)
        inner = torch.sum(inner, dim=-1)
        inter = self.l2loss(anchor, negative)
        inter = torch.sum(inter, dim=-1)

        loss = inner - inter + self.alpha

        return loss

