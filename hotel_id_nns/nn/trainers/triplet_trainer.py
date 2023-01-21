from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Union

import torch
from torch import nn

from hotel_id_nns.nn.trainers.trainer import Trainer
from hotel_id_nns.nn.losses.triplet_loss import TripletLoss
from hotel_id_nns.nn.datasets.h5_triplet_hotel_dataset import H5TripletHotelDataset


class ClassificationType(str, Enum):
    chain_id = 'chain-id'
    hotel_id = 'hotel-id'


class TripletTrainer(Trainer):

    class Config(Trainer.Config):

        def __init__(self, epochs: int, batch_size: int, learning_rate: float, weight_decay: float,
                     lr_patience: int, lr_cooldown: int, save_checkpoint: bool, amp: bool,
                     loss_type: str,
                     activate_wandb: bool, optimizer_name: str, load_from_model: Optional[Path],
                     dataloader_num_workers: Optional[int]):
            self.loss_type = loss_type
            super().__init__(epochs, batch_size, learning_rate, weight_decay, lr_patience,
                             lr_cooldown, save_checkpoint, amp, activate_wandb, optimizer_name,
                             load_from_model, dataloader_num_workers)
        @staticmethod
        def from_config(config: dict):
            parent_conf = Trainer.Config.from_config(config)
            return TripletTrainer.Config(**dict(parent_conf), loss_type=config['loss_type'])

    def __init__(
        self,
        project_name: str,
        trainer_id: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(project_name=project_name, trainer_id=trainer_id, device=device)

    def infer(
            self,
            net: nn.Module,
            batch: List[torch.Tensor],
            loss_criterion,
            compute_metrics: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:

        # TODO: Add metric computation

        # images
        for i in range(3):
            batch[i] = batch[i].to(device=self.device)

        # labels
        for i in range(3, len(batch)):
            batch[i] = torch.atleast_1d(batch[i].to(device=self.device).squeeze())

        a_imgs, p_imgs, n_imgs, a_chain_ids, p_chain_ids, n_chain_ids, a_hotel_ids, p_hotel_ids, n_hotel_ids = batch

        a_features = net(a_imgs)
        p_features = net(p_imgs)
        n_features = net(n_imgs)

        loss, metrics = loss_criterion(a_features, p_features, n_features)

        return loss, metrics

    def train(
        self,
        net: torch.nn.Module,
        config: Config,
        train_ds: H5TripletHotelDataset,
        checkpoint_dir: Path,
        val_ds: H5TripletHotelDataset,
    ):
        if config.loss_type == 'MSE':
            distance_function = torch.nn.PairwiseDistance(p=2)
        elif config.loss_type == 'Cosine':
            distance_function = lambda a,b: -torch.nn.CosineSimilarity(dim=1)(a,b)
        else:
            raise NotImplementedError()

        loss_criterion = TripletLoss(
            distance_function=distance_function,
            margin=0.2,
        )

        return super()._train(
            net=net,
            config=config,
            train_ds=train_ds,
            val_ds=val_ds,
            checkpoint_dir=checkpoint_dir,
            loss_criterion=loss_criterion,
        )

    def on_new_epoch(self):
        return super().on_new_epoch()
