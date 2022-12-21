from abc import abstractmethod
import logging
import os
from pathlib import Path
import random
import time
from typing import Optional, Union
from joblib import cpu_count
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader, Dataset
from hotel_id_nns.nn.losses.vae_loss import VAELoss
import wandb
from tqdm import tqdm

from hotel_id_nns.utils.pytorch import aggregate_metics, get_optimizer, load_model_weights
from hotel_id_nns.nn.datasets.triplet_sampler import TripletSampler
from hotel_id_nns.nn.datasets.triplet_hotel_dataset import TripletHotelDataset

ROOT_DIR = Path(__file__).parent.parent.parent.parent


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def get_loss_criterion(loss_type: str):
    # i -> input, t -> target, r -> region mask

    if loss_type == 'VAELoss':
        return VAELoss(kld_weight=0.2)

    elif loss_type == 'abs_l1_loss':
        return lambda i, t, r: nn.L1Loss(reduction='sum')(i * r, t * r) / len(i)

    elif loss_type == 'mean_l1_loss':
        return lambda i, t, r: nn.L1Loss(reduction='sum')(i * r, t * r) / torch.sum(r)

    elif loss_type == 'mean_l2_loss':
        return lambda i, t, r: nn.MSELoss(reduction='sum')(i * r, t * r) / torch.sum(r)

    elif loss_type == 'huber_loss':
        return lambda i, t, r: nn.HuberLoss(reduction='sum', delta=1)(i * r, t * r) / torch.sum(r)

    else:
        RuntimeError("loss function not given")


class Trainer:

    class Config:

        def __init__(
            self,
            epochs: int,
            batch_size: int,
            learning_rate: float,
            weight_decay: float,
            lr_patience: int,
            save_checkpoint: bool,
            amp: bool,
            activate_wandb: bool,
            optimizer_name: str,
            load_from_model: Optional[Path],
            dataloader_num_workers: Optional[int],
        ):
            self.epochs = epochs
            self.batch_size = batch_size
            self.learning_rate = learning_rate
            self.weight_decay = weight_decay
            self.lr_patience = lr_patience
            self.save_checkpoint = save_checkpoint
            self.amp = amp
            self.activate_wandb = activate_wandb
            self.optimizer_name = optimizer_name
            self.load_from_model = load_from_model
            self.dataloader_num_workers = dataloader_num_workers if not None else cpu_count()

        @staticmethod
        def from_config(config: dict):
            return Trainer.Config(
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                learning_rate=config['learning_rate'],
                weight_decay=config['weight_decay'],
                lr_patience=config['lr_patience'],
                save_checkpoint=config['save_checkpoint'],
                amp=config['amp'],
                activate_wandb=config['activate_wandb'],
                optimizer_name=config['optimizer_name'],
                load_from_model=Path(config['load_from_model'])
                if 'load_from_model' in config else None,
                dataloader_num_workers=config['dataloader_num_workers']
                if 'dataloader_num_workers' in config else None,
            )

        def __iter__(self):
            for attr, value in self.__dict__.items():
                yield attr, value

        def __repr__(self):
            return f"""
                Epochs:                 {self.epochs}
                Batch Size:             {self.batch_size}
                Learning Rate:          {self.learning_rate}
                Weight Decay:           {self.weight_decay}
                LR Patiance:            {self.lr_patience}
                Save Checkpoints:       {self.save_checkpoint}
                AMP:                    {self.amp}
                WandB:                  {self.activate_wandb}
                Optimizer Name:         {self.optimizer_name}
                Load From Model:        {self.load_from_model}
                Num Dataloader Worker:  {self.dataloader_num_workers}
            """

    def __init__(
        self,
        project_name: str,  # name for wandb
        trainer_id: Optional[str] = None,
        device: Optional[torch.device] = None,
        # dataset_config: BasicDataset.Config
    ):
        self.project_name = project_name
        if trainer_id is None:
            self.trainer_id = str(time.time())
        else:
            self.trainer_id = trainer_id

        self.device = device if device is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    @abstractmethod
    def infer(self, net, batch, loss_criterion, compute_metrics: bool = False):
        raise NotImplementedError()

    def evaluate(self, net: nn.Module, dataloader: DataLoader, loss_criterion):
        num_val_batches = len(dataloader)
        assert num_val_batches != 0, "at least one batch must be selected for evaluation"

        # iterate over the validation set
        infos = []
        loss = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Validation round', unit='batch', leave=False):
                batch_loss, info = self.infer(net, batch, loss_criterion, compute_metrics=True)
                loss += batch_loss
                infos.append(info)

        infosb = aggregate_metics(infos)
        if 'cm' in infosb:
            infosb.pop('cm')

        histograms = {}
        for tag, value in net.named_parameters():
            tag = tag.replace('/', '.')
            if value.grad is not None:  # if grad none, weights won't change
                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

        info = {**infosb, **histograms}

        return loss / num_val_batches, info

    @abstractmethod
    def train(
        self,
        net: torch.nn.Module,
        config: Config,
        # evaluation_dir: Path,
        train_ds: Dataset,
        val_ds: Dataset,
    ):
        raise NotImplementedError()

    def _train(
        self,
        net: torch.nn.Module,
        config: Config,
        checkpoint_dir: Path,
        train_ds: Dataset,
        val_ds: Dataset,
        loss_criterion,
    ):
        if config.save_checkpoint:
            dir_checkpoint = checkpoint_dir / self.project_name / self.trainer_id
            dir_checkpoint.mkdir(parents=True, exist_ok=True)

        n_train = len(train_ds)
        n_val = len(val_ds)

        # (Initialize logging)
        if config.activate_wandb:
            wandb.init(project=self.project_name,
                       resume='allow',
                       entity="hotel-id-nns",
                       reinit=True)
            wandb.config.update(
                dict(
                    # TODO:
                    # **dict(self.dataset_config),
                    # **dict(net.config),
                    **dict(config),
                    trainer_id=self.trainer_id,
                    training_size=n_train,
                    validation_size=n_val,
                ))

        logging.info(f'''Starting training:
            Training size:       {n_train}
            Validation size:     {n_val}
            Device:              {self.device.type}
            Trainer Config:
                {config}''')

        if config.load_from_model and str(config.load_from_model).lower() != 'none':
            net.load_state_dict(load_model_weights(ROOT_DIR / config.load_from_model))

        # not needed currently (only 1 gpu training on): net = nn.DataParallel(net)

        # allot network to run on multiple gpus
        net.to(self.device)

        if isinstance(train_ds, TripletHotelDataset):
            train_args = dict(shuffle=False, sampler=TripletSampler(train_ds))
        else:
            train_args = dict(shuffle=True)

        # create train and val data loader
        loader_args = dict(batch_size=config.batch_size, pin_memory=True)
        train_loader = DataLoader(train_ds,
                                  num_workers=config.dataloader_num_workers,
                                  **train_args,
                                  **loader_args)
        val_loader = DataLoader(val_ds,
                                shuffle=False,
                                num_workers=max(round(config.dataloader_num_workers * 0.5), 1),
                                **loader_args)

        # Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
        optimizer = get_optimizer(net=net,
                                  name=config.optimizer_name,
                                  weight_decay=config.weight_decay,
                                  learning_rate=config.learning_rate)

        # create learning rate reducer and gradient scaler to speedup convergency
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            'min',
            #            threshold=1e-1,
            cooldown=3,
            patience=config.lr_patience,
            verbose=True)
        # grad_scaler = torch.cuda.amp.GradScaler(enabled=config.amp)
        global_step = 0
        best_val_loss = torch.inf

        # Begin training
        for epoch in range(config.epochs):
            epoch_loss = 0
            with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{config.epochs}', unit='img') as pbar:
                iter_train_loder = iter(train_loader)
                net.train()

                for i in range(len(train_loader)):
                    batch = next(iter_train_loder)
                    optimizer.zero_grad()

                    with torch.cuda.amp.autocast(enabled=config.amp):
                        loss, _ = self.infer(net, batch, loss_criterion, compute_metrics=False)

                    assert not torch.isnan(loss)

                    # grad_scaler.scale(loss).backward()
                    # grad_scaler.step(optimizer)
                    # grad_scaler.update()
                    loss.backward()
                    optimizer.step()

                    pbar.update(config.batch_size)
                    global_step += 1
                    epoch_loss += loss.item()

                    # log infer to wandb
                    if config.activate_wandb:
                        wandb.log({
                            'step': global_step * config.batch_size,
                            'epoch': epoch,
                            'train loss': loss.item(),
                            'learning rate': optimizer.param_groups[0]['lr'],
                        })

                    pbar.set_postfix(**{'loss (batch)': loss.item()})

            net.eval()

            # validation round used to update learning rate
            logging.debug("evaluation round")
            # validation round to adapt learning rate
            epoch_val_loss, info = self.evaluate(net, val_loader, loss_criterion)
            lr_scheduler.step(epoch_val_loss)

            logging.info(
                f"lr info: num_bad_epochs {lr_scheduler.num_bad_epochs}, patience {lr_scheduler.patience}, cooldown {lr_scheduler.cooldown_counter} best {lr_scheduler.best}"
            )
            logging.info('Validation Loss: {}'.format(epoch_val_loss))

            if config.activate_wandb:
                wandb.log({
                    'step': global_step * config.batch_size,
                    'epoch': epoch,
                    **info,
                    'validation loss': epoch_val_loss,
                    'lr patience': lr_scheduler.num_bad_epochs / lr_scheduler.patience,
                })

            # save best epochs
            if best_val_loss > epoch_val_loss:
                best_val_loss = epoch_val_loss

                # save checkpoint if val loss decreased
                if config.save_checkpoint:
                    torch.save(net.state_dict(), str(dir_checkpoint / f'e{epoch+1}.pth'))
                    logging.info(f'Checkpoint {epoch + 1} saved!')

        if config.activate_wandb:
            wandb.finish()
