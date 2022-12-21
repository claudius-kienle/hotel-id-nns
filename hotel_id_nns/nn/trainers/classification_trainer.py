from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from matplotlib import pyplot as plt
from torch import nn

import torch
from hotel_id_nns.nn.datasets.h5_hotel_dataset import H5HotelDataset

from hotel_id_nns.nn.datasets.hotel_dataset import HotelDataset

from hotel_id_nns.nn.trainers.trainer import Trainer
from hotel_id_nns.utils.plotting import plot_confusion_matrix
from hotel_id_nns.utils.pytorch import compute_metrics
# from hotel_id_nns.utils.pytorch import get_accuracy

class ClassificationType(str, Enum):
    chain_id = 'chain-id'
    hotel_id = 'hotel-id'

class ClassificationTrainer(Trainer):
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
            return ClassificationTrainer.Config(**dict(parent_conf), loss_type=config['loss_type'])

    def __init__(
        self,
        classification_type: ClassificationType,
        trainer_id: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(project_name=classification_type.name, trainer_id=trainer_id, device=device)
        self.classification_type = classification_type
        self.verbose = False

    def infer(self,
              net: nn.Module,
              batch,
              loss_criterion,
              detailed_info: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """infers the classification net and computes the loss with the predicted class probs and the true class label

        Arguments:
            net: classification network that outputs probabilities given a batch of images
            batch: batch of inputs and chain id labels
            loss_criterion: loss function to compute loss with
            detailed_into: flag if returning object info should contain detailed information (will be true on evaluation round)


        Returns
            loss: computed loss
            info: dict containing debug information to display in wandb
        """
        input_img, chain_ids, hotel_ids = batch

        input_img = input_img.to(device=self.device)
        chain_ids = torch.atleast_1d(chain_ids.to(device=self.device).squeeze())
        hotel_ids = torch.atleast_1d(hotel_ids.to(device=self.device).squeeze())

        pred_probs = net(input_img)

        if self.classification_type == ClassificationType.chain_id:
            loss = loss_criterion(pred_probs, chain_ids)  # type: torch.Tensor
            metrics = compute_metrics(pred_probs, chain_ids)
        elif self.classification_type == ClassificationType.hotel_id:
            loss = loss_criterion(pred_probs, hotel_ids)  # type: torch.Tensor
            metrics = compute_metrics(pred_probs, hotel_ids)
        else:
            raise NotImplementedError()

        if self.verbose:
            ax = plt.subplot()
            plot_confusion_matrix(metrics['cm'], ax=ax)
            plt.savefig("cm.png")


        if not detailed_info:
            metrics.pop('cm')

        return loss, metrics

    def train(
        self,
        net: torch.nn.Module,
        config: Config,
        train_ds: Union[HotelDataset, H5HotelDataset],
        checkpoint_dir: Path,
        val_ds: Union[HotelDataset, H5HotelDataset],
    ):
        loss_type = config.loss_type

        if hasattr(train_ds, 'class_weights'): # backward compatible
            weights = train_ds.class_weights.to(self.device)
        elif self.classification_type == ClassificationType.chain_id:
            weights = train_ds.chain_id_weights.to(self.device)
        elif self.classification_type == ClassificationType.hotel_id:
            weights = train_ds.hotel_id_weights.to(self.device)
        else:
            raise NotImplementedError()

        if loss_type == 'NegativeLogLikelihood':
            loss_criterion = torch.nn.NLLLoss(reduction='mean', weight=weights)
        elif loss_type == 'CrossEntropy':
            loss_criterion = torch.nn.CrossEntropyLoss(reduction='mean', weight=weights)
        else:
            raise NotImplementedError()

        return super()._train(
            net=net,
            config=config,
            train_ds=train_ds,
            val_ds=val_ds,
            checkpoint_dir=checkpoint_dir,
            loss_criterion=loss_criterion,
        )
