from argparse import ArgumentParser
import torch
import json
import logging
from pathlib import Path
from typing import Dict
from torch import nn
import torchvision
from hotel_id_nns.nn.datasets.h5_triplet_hotel_dataset import H5TripletHotelDataset
from hotel_id_nns.nn.trainers.classification_trainer import ClassificationTrainer
from hotel_id_nns.nn.trainers.triplet_trainer import TripletTrainer
from hotel_id_nns.nn.modules.triplet_net import TripletNet
from hotel_id_nns.nn.modules.resnet_chris import ResNet, resnet18_cfg, resnet50_cfg
from hotel_id_nns.utils.pytorch import load_model_weights, inject_dropout

dir_path = Path(__file__).parent
repo_path = dir_path.parent.parent

def get_model(config: Dict, latent_size: int = 128) -> nn.Module:
    model_name = config['model_name']

    if model_name == 'ResNet18':
        backbone = ResNet(resnet18_cfg, out_features=latent_size)
    elif model_name == 'ResNet50':
        backbone = ResNet(resnet18_cfg, out_features=latent_size)
    else:
        raise NotImplementedError()
    
    return TripletNet(backbone)

def train(config: Dict, data_path: Path):
    print(config)
    ds_config = config['dataset']
    train_annotations = Path(data_path / ds_config['training'])
    train_ds = H5TripletHotelDataset(annotations_file_path=train_annotations, config=config['dataset'])
    val_annotations = Path(data_path / ds_config['validation'])
    val_ds = H5TripletHotelDataset(annotations_file_path=val_annotations, config=config['dataset'])

    checkpoint_dir = Path(repo_path / config['model_output'])

    trainer_config = TripletTrainer.Config.from_config(config['trainer'])

    trainer = TripletTrainer()
    
    class_net = get_model(config=config, latent_size=128)

    trainer.train(
        net=class_net,
        config=trainer_config,
        checkpoint_dir=checkpoint_dir,
        train_ds=train_ds,
        val_ds=val_ds,
    )
    


def main(args):
    config_file = repo_path / args.config_path
    assert config_file.suffix == '.json'

    with (config_file).open(mode="r") as f:
        config = json.load(f)

    if args.model is not None:
        print("overwriting model")
        config['model_name'] = args.model

    data_path = args.data_path if args.data_path is not None else repo_path

    train(config, data_path)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = ArgumentParser()
    parser.add_argument('config_path', type=Path, default='data/config/train_class_net.json')
    parser.add_argument('--data-path', type=Path, default=None)
    parser.add_argument('-m', '--model', type=str, default=None)
    main(parser.parse_args())