from argparse import ArgumentParser
import json
import logging
from pathlib import Path
import time
from hotel_id_nns.nn.datasets.chain_dataset import ChainDataset
from hotel_id_nns.nn.modules.VAE import VAE
from hotel_id_nns.nn.trainers.trainer import Trainer
from hotel_id_nns.nn.trainers.vae_trainer import VAETrainer

dir_path = Path(__file__).parent
repo_path = dir_path.parent.parent

def train_vae(args):
    input_size = 512

    # TODO: args
    train_annotations = Path(repo_path / "data/dataset/hotel_train_chain.csv")
    train_ds = ChainDataset(annotations_file_path=train_annotations, size=input_size)
    val_annotations = Path(repo_path / "data/dataset/hotel_val_chain.csv")
    val_ds = ChainDataset(annotations_file_path=val_annotations, size=input_size)

    checkpoint_dir = Path(repo_path / "data/checkpoints/vae")

    with (repo_path / "data/configs/train_vae.json").open(mode="r") as f:
        config = json.load(f)

    trainer_config = VAETrainer.Config.from_config(config['trainer'])


    # trainer_config = VAETrainer.Config(
    #     epochs=1,
    #     activate_wandb=False,
    #     amp=False,
    #     batch_size=1,
    #     learning_rate=1e-4,
    #     load_from_model=None,
    #     kld_loss_weight=0.2,
    #     lr_patience=100,
    #     optimizer_name='sgd',
    #     save_checkpoint=False,
    #     val_interval=1000,
    # )

    trainer = VAETrainer()

    vae = VAE(
        in_size=input_size,
        in_out_channels=3,
        hidden_channels=[32, 64, 128, 256],
        latent_dim=512,
    )

    trainer.train(
        net=vae,
        config=trainer_config,
        checkpoint_dir=checkpoint_dir,
        train_ds=train_ds,
        val_ds=val_ds,
    )
    


def main():
    logging.basicConfig(level=logging.INFO, force=True)
    parser = ArgumentParser()
    # TODO: parser.add_argument('config-path', type=Path)

    train_vae(parser.parse_args())

if __name__ == "__main__":
    main()