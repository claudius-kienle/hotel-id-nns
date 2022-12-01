import torch
from hotel_id_nns.nn.losses.bce_loss import BCELoss

from hotel_id_nns.nn.losses.kld_loss import KLDLoss


class VAELoss(object):

    def __init__(self, kld_weight: float) -> None:
        self.reconstruction_loss = BCELoss()
        self.kld_loss = KLDLoss()
        self.kld_weight = kld_weight


    def __call__(
        self,
        input: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        prediction: torch.Tensor,
    ):
        recon = self.reconstruction_loss(prediction, input)
        kl = self.kld_loss(mu=mu, logvar=logvar)

        loss = (1 - self.kld_weight) * recon + self.kld_weight * kl

        info = {
            'kld_loss': kl,
            'recon_loss': recon,
        }

        return loss, info
