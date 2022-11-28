from argparse import ArgumentParser
import json
import logging
from pathlib import Path
from hotel_id_nns.nn.datasets.chain_dataset import ChainDataset
from hotel_id_nns.nn.modules.ClassNet import ClassNet
from hotel_id_nns.nn.modules.VAE import VAE
from hotel_id_nns.nn.trainers.chain_id_trainer import ChainIDTrainer
from hotel_id_nns.nn.trainers.vae_trainer import VAETrainer

dir_path = Path(__file__).parent
repo_path = dir_path.parent.parent

def train_chain_id(args):
    with (repo_path / args.config_path).open(mode="r") as f:
        config = json.load(f)

    ds_config = config['dataset']
    train_annotations = Path(repo_path / ds_config['training'])
    train_ds = ChainDataset(annotations_file_path=train_annotations, config=config['dataset'])
    val_annotations = Path(repo_path / ds_config['validation'])
    val_ds = ChainDataset(annotations_file_path=val_annotations, config=config['dataset'])

    checkpoint_dir = Path(repo_path / config['model_output'])

    trainer_config = ChainIDTrainer.Config.from_config(config['trainer'])

    trainer = ChainIDTrainer()

    class_net = ClassNet(
        in_size=ds_config['input_size'],
        in_channels=3,
        hidden_channels=[32, 64, 128, 256],
        num_classes=ds_config['num_chain_id_classes']
    )

    trainer.train(
        net=class_net,
        config=trainer_config,
        checkpoint_dir=checkpoint_dir,
        train_ds=train_ds,
        val_ds=val_ds,
    )
    


def main():
    logging.basicConfig(level=logging.INFO, force=True)
    parser = ArgumentParser()
    parser.add_argument('config_path', type=Path, default='data/config/train_class_net.json')

    train_chain_id(parser.parse_args())

if __name__ == "__main__":
    main()