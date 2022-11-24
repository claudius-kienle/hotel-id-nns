import torch
from torch import nn

from hotel_id_nns.nn.losses.KLDLoss import KLDLoss


class VAELoss2(object):

    def __init__(self, kld_weight: float) -> None:
        self.reconstruction_loss = nn.MSELoss()
        self.kld_loss = KLDLoss()
        self.kld_weight = kld_weight


    def __call__(
        self,
        input: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        prediction: torch.Tensor,
    ):
        _, _, w, h = input.shape
        pixels = w * h
        recon = torch.nn.functional.binary_cross_entropy(input=prediction, target=input) * pixels

        std = torch.exp(0.5 * logvar)
        kl = 1 + std - mu ** 2 - torch.exp(std)
        kl = torch.sum(kl, dim=-1)
        kl = kl * -.05

        loss = (1 - self.kld_weight) * recon + self.kld_weight * kl

        info = {
            'kld_loss': kl,
            'recon_loss': recon,
        }

        return loss, info
