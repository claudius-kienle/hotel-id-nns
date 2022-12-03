from pathlib import Path
from typing import Optional, Tuple
from torch import nn

import torch
from torch.utils.data import Dataset
import wandb
from hotel_id_nns.nn.trainers.trainer import Trainer


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
            return ChainIDTrainer.Config(
                **dict(parent_conf),
                loss_type=config['loss_type']
            )
        

    def __init__(
        self,
        trainer_id: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(trainer_id, device)

    def infer(self, net: nn.Module, batch, loss_criterion, detailed_info: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
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
        input_img, chain_id = batch

        input_img = input_img.to(device=self.device)
        chain_id = chain_id.to(device=self.device)

        pred_chain_id_probs = net(input_img)
        num_classes = pred_chain_id_probs.shape[-1]
        pred_chain_id = torch.argmax(pred_chain_id_probs, dim=-1)

        loss = loss_criterion(pred_chain_id_probs, chain_id)

        indices = num_classes * chain_id + pred_chain_id
        cm = torch.bincount(indices, minlength=num_classes ** 2).reshape((num_classes, num_classes))

        accuracy = cm.diag().sum() / (cm.sum() + 1e-15)
        precision = cm.diag() / (cm.sum(dim=0) + 1e-15)
        recall = cm.diag() / (cm.sum(dim=1) + 1e-15)
        f1 = 2 * precision * recall / (precision + recall + 1e-15)

        info = {
            'accuracy': accuracy,
            'precision':precision.mean(), 
            'recall': recall.mean(),
            'f1': f1.mean(),
        }

        if detailed_info:
            info['input'] = wandb.Image(input_img.cpu().detach().squeeze().numpy().transpose((1, 2, 0)))

        return loss, info

    def train(
        self,
        net: torch.nn.Module,
        config: Config,
        train_ds: Dataset,
        checkpoint_dir: Path,
        val_ds: Dataset,
    ):
        loss_type = config.loss_type
        if loss_type == 'NegativeLogLikelihood':
            loss_criterion = torch.nn.NLLLoss(reduction='mean')
        elif loss_type == 'CrossEntropy':
            loss_criterion = torch.nn.CrossEntropyLoss(reduction='mean')
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
