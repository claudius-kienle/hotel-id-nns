from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset
from hotel_id_nns.nn.losses.VAELoss import VAELoss
from hotel_id_nns.nn.losses.vae_loss_2 import VAELoss2
from hotel_id_nns.nn.modules.VAE import VAE
from hotel_id_nns.nn.trainers.trainer import Trainer


class ChainIDTrainer(Trainer):

    class Config(Trainer.Config):

        def __init__(
            self,
            **kwargs,
        ):
            super().__init__(**kwargs)

        @staticmethod
        def from_config(config: dict):
            parent_conf = Trainer.Config.from_config(config)
            return ChainIDTrainer.Config(
                **dict(parent_conf),
            )
        

    def __init__(
        self,
        trainer_id: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(trainer_id, device)

    def infer(self, net: VAE, batch, loss_criterion) -> Tuple[torch.Tensor, torch.Tensor]:
        input_img, chain_id = batch

        input_img = input_img.to(device=self.device)

        pred_chain_id = net(input_img)

        loss = loss_criterion(pred_chain_id, chain_id)

        info = {
            'output': {
                'pred': pred_chain_id,
                'true': chain_id
            }
        }

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
        loss_criterion = torch.nn.NLLLoss()
        return super()._train(
            net=net,
            config=config,
            train_ds=train_ds,
            val_ds=val_ds,
            checkpoint_dir=checkpoint_dir,
            loss_criterion=loss_criterion,
        )
