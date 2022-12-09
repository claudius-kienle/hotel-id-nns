from itertools import chain
from pathlib import Path
from typing import Dict, Optional, Tuple
from matplotlib import pyplot as plt
from torch import nn

import torch
import wandb
from wandb.plot import confusion_matrix
from hotel_id_nns.nn.datasets.chain_dataset import ChainDataset
from hotel_id_nns.nn.trainers.trainer import Trainer
from hotel_id_nns.utils.plotting import plot_confusion_matrix
from hotel_id_nns.utils.pytorch import compute_metrics
# from hotel_id_nns.utils.pytorch import get_accuracy


class ChainIDTrainer(Trainer):
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
            return ChainIDTrainer.Config(**dict(parent_conf), loss_type=config['loss_type'])

    def __init__(
        self,
        trainer_id: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(trainer_id, device)
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
        input_img, chain_ids = batch

        input_img = input_img.to(device=self.device)
        chain_ids = torch.atleast_1d(chain_ids.to(device=self.device).squeeze())

        pred_chain_id_probs = net(input_img)

        loss = loss_criterion(pred_chain_id_probs, chain_ids)  # type: torch.Tensor

        metrics = compute_metrics(pred_chain_id_probs, chain_ids)

        if self.verbose:
            ax = plt.subplot()
            plot_confusion_matrix(metrics['cm'], ax=ax)
            plt.savefig("cm.png")

        info = {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
        }

        # if detailed_info:
        info['cm'] = confusion_matrix(pred_chain_id_probs.cpu().detach().numpy(), chain_ids.cpu().detach().tolist(), class_names=list(map(str, range(pred_chain_id_probs.shape[1]))))

        return loss, info

    def train(
        self,
        net: torch.nn.Module,
        config: Config,
        train_ds: ChainDataset,
        checkpoint_dir: Path,
        val_ds: ChainDataset,
    ):
        loss_type = config.loss_type
        chain_id_weights = train_ds.chain_id_weights.to(self.device)
        if loss_type == 'NegativeLogLikelihood':
            loss_criterion = torch.nn.NLLLoss(reduction='mean', weight=chain_id_weights)
        elif loss_type == 'CrossEntropy':
            loss_criterion = torch.nn.CrossEntropyLoss(reduction='mean', weight=chain_id_weights)
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
