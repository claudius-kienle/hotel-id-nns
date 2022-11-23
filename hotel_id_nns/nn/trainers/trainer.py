from abc import abstractmethod
import logging
from pathlib import Path
import random
import time
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from hotel_id_nns.nn.losses.VAELoss import VAELoss
import wandb
from tqdm import tqdm


ROOT_DIR = Path(__file__).parent.parent.parent


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
                lr_patience: int,
                val_interval: int,
                save_checkpoint: bool,
                amp: bool,
                activate_wandb: bool,
                optimizer_name: str,
                load_from_model: Optional[Path],
        ):
            self.epochs = epochs
            self.batch_size = batch_size
            self.learning_rate = learning_rate
            self.lr_patience = lr_patience
            self.val_interval = val_interval
            self.save_checkpoint = save_checkpoint
            self.amp = amp
            self.activate_wandb = activate_wandb
            self.optimizer_name = optimizer_name
            self.load_from_model = load_from_model

        @staticmethod
        def from_config(config: dict):
            return Trainer.Config(
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                learning_rate=config['learning_rate'],
                lr_patience=config['lr_patience'],
                save_checkpoint=config['save_checkpoint'],
                amp=config['amp'],
                activate_wandb=config['activate_wandb'],
                val_interval=config['val_interval'],
                optimizer_name=config['optimizer_name'],
                load_from_model=Path(config['load_from_model']) if config['load_from_model'] != "None" else None
            )

        def __iter__(self):
            for attr, value in self.__dict__.items():
                yield attr, value

        def __repr__(self):
            return f"""
                Epochs:              {self.epochs}
                Batch Size:          {self.batch_size}
                Learning Rate:       {self.learning_rate}
                LR Patiance:         {self.lr_patience}
                Save Checkpoints:    {self.save_checkpoint}
                AMP:                 {self.amp}
                WandB:               {self.activate_wandb}
                Validation Interval  {self.val_interval}
                Optimizer Name:      {self.optimizer_name}
                Load From Model:      {self.load_from_model}
            """

    def __init__(
        self,
        trainer_id: Optional[str] = None,
        device: Optional[torch.device] = None,
        # dataset_config: BasicDataset.Config
    ):
        if trainer_id is None:
            self.trainer_id = str(time.time())
        else:
            self.trainer_id = trainer_id

        self.device = device if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        # self.dataset_config = dataset_config

    @abstractmethod
    def infer( self, net, batch, loss_criterion):
        raise NotImplementedError()

    def evaluate(
        self,
        net: nn.Module,
        dataloader: DataLoader,
        loss_criterion
    ):
        net.eval()
        num_val_batches = len(dataloader)
        assert num_val_batches != 0, "at least one batch must be selected for evaluation"

        loss = 0

        # iterate over the validation set
        for batch in tqdm(dataloader, desc='Validation round', unit='batch', leave=False):
            with torch.no_grad():
                _, batch_loss = self.infer(net, batch, loss_criterion)
                loss += batch_loss

        net.train()

        return loss / num_val_batches

    def __evaluate_for_visualization(self, dataloader: DataLoader, net, loss_criterion):
        idx = random.randint(0, len(dataloader) - 1)
        batch = list(iter(dataloader))[idx]

        predictions, _ = self.infer(net, batch, loss_criterion)

        histograms = {}
        for tag, value in net.named_parameters():
            tag = tag.replace('/', '.')
            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

        # visualizate images
        vis_image = batch['image'][0].numpy().transpose((1, 2, 0))
        nan_mask = batch['nan-mask'][0].numpy().squeeze()
        region_mask = batch['region-mask'][0].numpy().squeeze()
        vis_depth_input = vis_image[..., 3].squeeze()
        vis_depth_label = batch['label'][0].numpy().squeeze()
        depth_prediction = predictions[0].float().cpu().detach().numpy().squeeze()

        # apply masks
        mask = np.logical_and(nan_mask, region_mask)
        vis_depth_prediction = np.where(mask, depth_prediction, np.nan)
        vis_depth_input = np.where(mask, vis_depth_input, np.nan)
        vis_depth_label = np.where(mask, vis_depth_label, np.nan)

        # undo normalization
        if self.dataset_config.normalize_depths:
            params = dict(min=self.dataset_config.normalize_depths_min, max=self.dataset_config.normalize_depths_max)
            vis_depth_input = unnormalize_depth(vis_depth_input, **params)
            vis_depth_label = unnormalize_depth(vis_depth_label, **params)
            vis_depth_prediction = unnormalize_depth(vis_depth_prediction, **params)

        return {
            'input': {
                'rgb': wandb.Image(to_rgb(vis_image[..., :3])),
                'depth': wandb.Image(visualize_depth(vis_depth_input)),
            },
            'output': {
                'label': wandb.Image(visualize_depth(vis_depth_label)),
                'pred': wandb.Image(visualize_depth(vis_depth_prediction)),
            },
            'masks': {
                'nan-mask': wandb.Image(visualize_mask(nan_mask)),
                'region-mask': wandb.Image(visualize_mask(region_mask)),
            },
            **histograms
        }

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
            dir_checkpoint = checkpoint_dir / f"{net.name}"

        division_step = (config.val_interval // config.batch_size)
        n_train = len(train_ds)
        n_val = len(val_ds)

        # (Initialize logging)
        if config.activate_wandb:
            experiment = wandb.init(project=net.name, resume='allow',
                                    entity="hotel-id-nns", reinit=True)
            experiment.config.update(
                dict(
                 # TODO:
                    # **dict(self.dataset_config),
                    # **dict(config),
                    # **dict(net.config),
                    trainer_id=self.trainer_id,
                    training_size=n_train,
                    validation_size=n_val,
                    evaluation_interval=division_step
                )
            )

        logging.info(f'''Starting training:
            Training size:       {n_train}
            Validation size:     {n_val}
            Device:              {self.device.type}
            Trainer Config:
                {config}'''
        )

        """logging.info(f'''Starting training:
            Training size:       {n_train}
            Validation size:     {n_val}
            Device:              {self.device.type}
            Network Config:
                {net.config.get_printout()}
            Dataset Config:      
                {self.dataset_config.get_printout()}
            Trainer Config:
                {config.get_printout()}
            ''')"""

        if config.load_from_model:
            net.load_state_dict(torch.load(ROOT_DIR / config.load_from_model, map_location='cpu'))

        # net = nn.DataParallel(net)
        net.to(self.device)

        loader_args = dict(
            batch_size=config.batch_size,
            num_workers=4,
            pin_memory=True
        )
        train_loader = DataLoader(
            train_ds,
            shuffle=True,
            **loader_args
        )
        val_loader = DataLoader(
            val_ds,
            shuffle=False,
            **loader_args
        )

        # Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
        if config.optimizer_name == 'rmsprop':
            optimizer = optim.RMSprop(
                net.parameters(),
                lr=config.learning_rate,
                weight_decay=1e-8,
                momentum=0.9
            )
        elif config.optimizer_name == 'adam':
            optimizer = optim.Adam(
                net.parameters(),
                lr=config.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0,
                amsgrad=False
            )
        elif config.optimizer_name == 'sgd':
            optimizer = optim.SGD(
                net.parameters(),
                lr=config.learning_rate,
            )
        else:
            RuntimeError(f"invalid optimizer name given {config.optimizer_name}")

        lr_updates_per_epoch = max(n_train // 2000, 1)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            'min',
            #            threshold=1e-1,
            cooldown=3,
            patience=config.lr_patience * lr_updates_per_epoch,
            verbose=True
        )
        grad_scaler = torch.cuda.amp.GradScaler(enabled=config.amp)
        global_step = 0

        # Begin training
        for epoch in range(config.epochs):
            net.train()
            epoch_loss = 0
            with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{config.epochs}', unit='img') as pbar:
               for batch in train_loader:
                    optimizer.zero_grad(set_to_none=True)

                    with torch.cuda.amp.autocast(enabled=config.amp):
                        _, loss = self.infer(net, batch, loss_criterion)

                    assert not torch.isnan(loss)

                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(optimizer)
                    grad_scaler.update()

                    pbar.update(config.batch_size)
                    global_step += 1
                    epoch_loss += loss.item()

                    if config.activate_wandb:
                        experiment.log({
                            'step': global_step * config.batch_size,
                            'epoch': epoch,
                            'train loss': loss.item(),
                            'learning rate': optimizer.param_groups[0]['lr'],
                        })

                    pbar.set_postfix(**{'loss (batch)': loss.item()})

                    # Visualization round
                    # if global_step % division_step == 0 and config.activate_wandb:
                    #     logging.debug("visualization round")
                    #     experiment_log = self.__evaluate_for_visualization(val_loader, net, loss_criterion)
                    #     experiment_log['step'] = global_step * config.batch_size,
                    #     experiment_log['epoch'] = epoch
                    #     experiment.log(experiment_log)

                    if global_step % (n_train // (config.batch_size * lr_updates_per_epoch)) == 0:
                        logging.debug("evaluation round")
                        # validation round to adapt learning rate
                        epoch_val_loss = self.evaluate(net, val_loader, loss_criterion)
                        lr_scheduler.step(epoch_val_loss)
                        logging.info(
                            f"lr info: num_bad_epochs {lr_scheduler.num_bad_epochs}, patience {lr_scheduler.patience}, cooldown {lr_scheduler.cooldown_counter} best {lr_scheduler.best}")
                        logging.info('Validation Loss: {}'.format(epoch_val_loss))

                        if config.activate_wandb:
                            experiment.log({
                                'step': global_step * config.batch_size,
                                'epoch': epoch,
                                'validation loss': epoch_val_loss,
                                'lr patience': lr_scheduler.num_bad_epochs / lr_scheduler.patience
                            })

            if config.save_checkpoint:
                dir_checkpoint.mkdir(parents=True, exist_ok=True)
                torch.save(net.state_dict(), str(dir_checkpoint / f'e{epoch+1}.pth'))
                logging.info(f'Checkpoint {epoch + 1} saved!')

        if config.activate_wandb:
            wandb.finish()
