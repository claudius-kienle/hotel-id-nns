from argparse import ArgumentParser
import json
import logging
from pathlib import Path
from typing import OrderedDict
import torch
import torchvision
from hotel_id_nns.nn.datasets.chain_dataset import ChainDataset
from hotel_id_nns.nn.datasets.chain_dataset_h5 import H5ChainDataset
from hotel_id_nns.nn.datasets.dataset_factory import DatasetFactory
from hotel_id_nns.nn.modules.resnet_chris import ResNet, resnet18_cfg
from hotel_id_nns.nn.trainers.chain_id_trainer import ChainIDTrainer

dir_path = Path(__file__).parent
repo_path = dir_path.parent.parent


def train_chain_id(args):
    config_file = repo_path / args.config_path
    assert config_file.suffix == '.json'

    with (config_file).open(mode="r") as f:
        config = json.load(f)

    ds_config = config['dataset']
    data_path = args.data_path if args.data_path is not None else repo_path
    train_annotations = Path(data_path / ds_config['training'])
    train_ds = DatasetFactory().get(train_annotations, config=config['dataset'])
    val_annotations = Path(data_path / ds_config['validation'])
    val_ds = DatasetFactory().get(val_annotations, config=config['dataset'])

    checkpoint_dir = Path(repo_path / config['model_output'])

    trainer_config = ChainIDTrainer.Config.from_config(config['trainer'])

    trainer = ChainIDTrainer()

    net = ResNet(resnet18_cfg, train_ds.num_chain_id_classes)

    trainer.train(
        net=net,
        config=trainer_config,
        checkpoint_dir=checkpoint_dir,
        train_ds=train_ds,
        val_ds=val_ds,
    )


def main():
    logging.basicConfig(level=logging.INFO, force=True)
    parser = ArgumentParser()
    parser.add_argument('config_path', type=Path, default='data/config/train_resnet_chris.json')
    parser.add_argument('--data-path', type=Path, default=None)

    train_chain_id(parser.parse_args())


if __name__ == "__main__":
    main()