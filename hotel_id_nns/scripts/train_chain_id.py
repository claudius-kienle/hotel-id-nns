from argparse import ArgumentParser
import json
import logging
from pathlib import Path
from typing import Dict
import torch
from torch import nn
import torchvision
from hotel_id_nns.nn.datasets.hotel_dataset import HotelDataset
from hotel_id_nns.nn.datasets.chain_dataset_h5 import H5ChainDataset
from hotel_id_nns.nn.datasets.dataset_factory import DatasetFactory
from hotel_id_nns.nn.modules.class_net import ClassNet
from hotel_id_nns.nn.trainers.chain_id_trainer import ChainIDTrainer

dir_path = Path(__file__).parent
repo_path = dir_path.parent.parent

def get_model(config: Dict, num_classes: int) -> nn.Module:
    model_name = config['model_name']
    use_weights = config['model_weights_imagenet']
    if model_name == 'ResNet18':
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if use_weights else None
        model = torchvision.models.resnet18(weights=weights)
    elif model_name == 'ResNet34':
        weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if use_weights else None
        model = torchvision.models.resnet34(weights=weights)
    elif model_name == 'ResNet50':
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if use_weights else None
        model = torchvision.models.resnet50(weights=weights) # weights=weights) # ,num_classes=
    else:
        raise NotImplementedError()

    if config['model_finetune']: # only finetune on final fc
        for param in model.parameters():
            param.requires_grad = False

    model.name = "ClassNet"
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    return model


def train_chain_id(config: Dict, data_path: Path):
    ds_config = config['dataset']
    train_annotations = Path(data_path / ds_config['training'])
    train_ds = DatasetFactory().get(train_annotations, config=config['dataset'])
    val_annotations = Path(data_path / ds_config['validation'])
    val_ds = DatasetFactory().get(val_annotations, config=config['dataset'])

    checkpoint_dir = Path(repo_path / config['model_output'])

    trainer_config = ChainIDTrainer.Config.from_config(config['trainer'])

    trainer = ChainIDTrainer()

    class_net = ClassNet(
        in_size=ds_config['input_size'],
        in_channels=3,
        hidden_channels=[32, 64, 128, 256, 512],
        num_classes=train_ds.num_classes
    )

    class_net = get_model(config, num_classes = train_ds.num_chain_id_classes)

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

    data_path = args.data_path if args.data_path is not None else repo_path

    train_chain_id(config, data_path)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = ArgumentParser()
    parser.add_argument('config_path', type=Path, default='data/config/train_class_net.json')
    parser.add_argument('--data-path', type=Path, default=None)
    main(parser.parse_args())