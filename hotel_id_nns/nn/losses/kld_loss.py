import torch


class KLDLoss(object):

    def __init__(self) -> None:
        pass

    def __call__(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        kl = 1 + std - mu ** 2 - torch.exp(std)
        kl = torch.sum(kl)
        kl = kl * -.5

        return kl