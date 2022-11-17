import torch
from torch import nn

from hotel_id_nns.nn.losses.KLDLoss import KLDLoss


class VAELoss(object):

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
        recon_loss = self.reconstruction_loss(input, prediction)
        kld_loss = self.kld_loss(mu=mu, logvar=logvar)
        loss = recon_loss + self.kld_weight * kld_loss 
        return loss
