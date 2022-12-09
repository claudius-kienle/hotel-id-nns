from pathlib import Path
from typing import Optional, Tuple
from torch import nn

import torch
from torch.utils.data import Dataset
import wandb
from hotel_id_nns.nn.datasets.chain_dataset import ChainDataset
from hotel_id_nns.nn.trainers.trainer import Trainer
from hotel_id_nns.utils.plotting import plot_confusion_matrix
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
        self.verbose = False

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
        chain_id = torch.atleast_1d(chain_id.to(device=self.device).squeeze())

        pred_chain_id_probs = net(input_img)
        num_classes = pred_chain_id_probs.shape[-1]
        pred_chain_id = torch.argmax(pred_chain_id_probs, dim=-1) 

        loss = loss_criterion(pred_chain_id_probs, chain_id)

        # print("pred_probs:", pred_chain_id_probs[0])
        print("label:", chain_id)
        print("preds:", pred_chain_id)
        # print("loss:", loss)

        # acc1, acc5 = get_accuracy(pred_chain_id_probs, chain_id, topk=(1, 5))

        indices = num_classes * chain_id + pred_chain_id
        cm = torch.bincount(indices, minlength=num_classes ** 2).reshape((num_classes, num_classes))
        
        if self.verbose:
            plot_confusion_matrix(cm)

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
