from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset
from hotel_id_nns.nn.losses.VAELoss import VAELoss
from hotel_id_nns.nn.modules.VAE import VAE
from hotel_id_nns.nn.trainers.trainer import Trainer


class VAETrainer(Trainer):
    class Config(Trainer.Config):
        def __init__(
            self,
            epochs: int,
            batch_size: int,
            learning_rate: float,
            lr_patience: int,
            val_interval: int,
            kld_loss_weight: float,
            save_checkpoint: bool,
            amp: bool,
            activate_wandb: bool,
            optimizer_name: str,
            load_from_model: Optional[Path],
        ):
            super().__init__(
                epochs,
                batch_size,
                learning_rate,
                lr_patience,
                val_interval,
                save_checkpoint,
                amp,
                activate_wandb,
                optimizer_name,
                load_from_model,
            )
            self.kld_loss_weight = kld_loss_weight

    def __init__(
        self,
        trainer_id: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(trainer_id, device)

    def infer(self, net: VAE, batch, loss_criterion: VAELoss) -> Tuple[torch.Tensor, torch.Tensor]:
        input_img, chain_id = batch

        input_img = input_img.to(device=self.device)

        output_img, mu, logvar = net(input_img)

        loss = loss_criterion(prediction=output_img, input=input_img, mu=mu, logvar=logvar)

        return output_img, loss

    def train(
        self,
        net: torch.nn.Module,
        config: Config,
        train_ds: Dataset,
        val_ds: Dataset,
    ):
        loss_criterion = VAELoss(kld_weight=config.kld_loss_weight)
        return super()._train(net, config, train_ds, val_ds, loss_criterion=loss_criterion)