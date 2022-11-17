import torch


class KLDLoss(object):

    def __init__(self) -> None:
        pass

    def __call__(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        loss = torch.mean(
            -0.5 * torch.sum(
                1 + logvar - mu**2 - torch.exp(logvar),
                dim=1,
            ),
            dim=0,
        )
        return loss