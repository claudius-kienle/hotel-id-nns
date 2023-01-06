from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Union

import torch
from torch import nn

from hotel_id_nns.nn.trainers.trainer import Trainer
from hotel_id_nns.nn.datasets.h5_triplet_hotel_dataset import H5TripletHotelDataset


class ClassificationType(str, Enum):
    chain_id = 'chain-id'
    hotel_id = 'hotel-id'


class TripletTrainer(Trainer):
    class Config(Trainer.Config):
        def __init__(
                self,
                loss_type: str,
                **kwargs,
        ):
            super().__init__(**kwargs)
            self.loss_type = loss_type

        @staticmethod
        def from_config(config: dict):
            parent_conf = Trainer.Config.from_config(config)
            return TripletTrainer.Config(**dict(parent_conf), loss_type=config['loss_type'])

    def __init__(
            self,
            classification_type: ClassificationType,
            trainer_id: Optional[str] = None,
            device: Optional[torch.device] = None,
    ):
        super().__init__(project_name=classification_type.value, trainer_id=trainer_id, device=device)
        self.classification_type = classification_type

    def infer(self,
              net: nn.Module,
              batch: List[torch.Tensor],
              loss_criterion,
              compute_metrics: bool = False) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:

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

        metrics = None

        loss: torch.Tensor = loss_criterion(a_features, p_features, n_features)

        return loss, metrics

    def train(
        self,
        net: torch.nn.Module,
        config: Config,
        train_ds: H5TripletHotelDataset,
        checkpoint_dir: Path,
        val_ds: H5TripletHotelDataset,
    ):
        # TODO: Add loss criterion here
        loss_criterion = None

        return super()._train(
            net=net,
            config=config,
            train_ds=train_ds,
            val_ds=val_ds,
            checkpoint_dir=checkpoint_dir,
            loss_criterion=loss_criterion,
        )
