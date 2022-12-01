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

        pred_chain_id = net(input_img)

        if isinstance(loss_criterion, torch.nn.NLLLoss):
            loss = loss_criterion(pred_chain_id, chain_id)
        elif isinstance(loss_criterion, torch.nn.BCEWithLogitsLoss):
            num_classes = pred_chain_id.shape[-1]
            loss = loss_criterion(pred_chain_id, torch.nn.functional.one_hot(chain_id, num_classes=num_classes).to(torch.float32))
        else:
            raise NotImplementedError()

        accuracy = (torch.argmax(pred_chain_id, dim=-1) == chain_id).sum() / len(chain_id)

        info = {
            'accuary': accuracy,
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
        if loss_type == 'NNLoss':
            loss_criterion = torch.nn.NLLLoss()
        elif loss_type == 'BCELoss':
            loss_criterion = torch.nn.BCEWithLogitsLoss()

        return super()._train(
            net=net,
            config=config,
            train_ds=train_ds,
            val_ds=val_ds,
            checkpoint_dir=checkpoint_dir,
            loss_criterion=loss_criterion,
        )
