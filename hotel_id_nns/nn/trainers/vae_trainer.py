from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset
import wandb
from hotel_id_nns.nn.losses.vae_loss import VAELoss
from hotel_id_nns.nn.modules.vae import VAE
from hotel_id_nns.nn.trainers.trainer import Trainer


class VAETrainer(Trainer):

    class Config(Trainer.Config):

        def __init__(
            self,
            kld_loss_weight: float,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self.kld_loss_weight = kld_loss_weight

        @staticmethod
        def from_config(config: dict):
            parent_conf = Trainer.Config.from_config(config)
            return VAETrainer.Config(
                **dict(parent_conf),
                kld_loss_weight=config['kld_loss_weight']
            )
        

    def __init__(
        self,
        trainer_id: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(trainer_id, device)

    def infer(self, net: VAE, batch, loss_criterion: VAELoss, detailed_info: bool = False) -> Tuple[torch.Tensor, Dict]:
        input_img, chain_id = batch

        input_img = input_img.to(device=self.device)

        output_img, mu, logvar = net(input_img)

        # from matplotlib import pyplot as plt
        # _, ax = plt.subplots(2,1)
        # ax[0].imshow(input_img[0].transpose(0, 2))
        # ax[1].imshow(output_img[0].transpose(0, 2).detach())
        # plt.show()

        loss, info = loss_criterion(prediction=output_img, input=input_img, mu=mu, logvar=logvar)

        if 'output' not in info:
            info['output'] = {}
        info['output']['pred'] = wandb.Image(output_img.cpu().detach().squeeze().numpy().transpose((1, 2, 0)))

        return loss, info

    def train(
        self,
        net: torch.nn.Module,
        config: Config,
        train_ds: Dataset,
        checkpoint_dir: Path,
        val_ds: Dataset,
    ):
        # loss_criterion = VAELoss(kld_weight=config.kld_loss_weight)
        loss_criterion = VAELoss(kld_weight=config.kld_loss_weight)
        return super()._train(
            net=net,
            config=config,
            train_ds=train_ds,
            val_ds=val_ds,
            checkpoint_dir=checkpoint_dir,
            loss_criterion=loss_criterion,
        )
