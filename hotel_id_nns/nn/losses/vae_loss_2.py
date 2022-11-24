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
        recon = torch.nn.functional.binary_cross_entropy(input=prediction, target=input) * input.shape[:2]

        std = torch.exp(0.5 * logvar)
        kl = 1 + std - mu ** 2 - torch.exp(std)
        kl = torch.sum(kl, dim=-1)
        kl = kl * -.05

        loss = torch.mean(recon + kl)

        info = {
            'kld_loss': kl,
            'recon_loss': recon,
        }

        return loss, info
