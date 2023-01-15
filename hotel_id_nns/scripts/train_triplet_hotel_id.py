from argparse import ArgumentParser
import torch
import json
import logging
from pathlib import Path
from typing import Dict
from torch import nn
import torchvision
from hotel_id_nns.nn.datasets.h5_triplet_hotel_dataset import H5TripletHotelDataset
from hotel_id_nns.nn.modules.resnet_chris import ResNet, resnet18_cfg, resnet50_cfg
from hotel_id_nns.nn.trainers.classification_trainer import ClassificationTrainer
from hotel_id_nns.nn.trainers.triplet_trainer import TripletTrainer
from hotel_id_nns.utils.pytorch import load_model_weights, inject_dropout

dir_path = Path(__file__).parent
repo_path = dir_path.parent.parent

def get_imagenet_weights(model_name: str):
    return load_model_weights(repo_path / ("data/checkpoints/image-net/%s-imagenet-weights.pth" % model_name.lower()))


def get_model(config: Dict, num_classes: int) -> nn.Module:
    model_name = config['model_name']
    use_weights = config['model_weights_imagenet']
    if model_name == 'ResNet18':
        model = ResNet(network_cfg=resnet18_cfg, out_features=num_classes)
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if use_weights else None
        model.load_state_dict(weights)
        # model = torchvision.models.resnet18(weights=weights)
    elif model_name == 'ResNet34':
        model = ResNet34(num_classes=num_classes)
        # weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if use_weights else None
        # model = torchvision.models.resnet34(weights=weights)
        # model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'ResNet50':
        model = ResNet(network_cfg=resnet50_cfg,out_features=num_classes)
    elif model_name == 'ResNet50-J':
        from hotel_id_nns.nn.modules.resnet_johannes import ResNet50
        model = ResNet50(num_classes=num_classes)
    elif model_name == 'ResNet50-T':
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if use_weights else None
        model = torchvision.models.resnet50(weights=weights)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'ResNet101-T':
        # model = ResNet101(num_classes=num_classes)
        weights = torchvision.models.ResNet101_Weights.IMAGENET1K_V2 if use_weights else None
        model = torchvision.models.resnet101(weights=weights) # weights=weights) # ,num_classes=
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'ResNet152-T':
        # model = ResNet152(num_classes=num_classes)
        weights = torchvision.models.ResNet152_Weights.IMAGENET1K_V2 if use_weights else None
        model = torchvision.models.resnet152(weights=weights) # weights=weights) # ,num_classes=
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        raise NotImplementedError()

    if 'dropout_rate' in config and config['dropout_rate'] != 0:
        inject_dropout(model, config['dropout_rate'])

    # summary(model, input_size=(3, 224, 224))

    # TODO: if config['model_finetune']: # only finetune on final fc
    # TODO:     for param in model.parameters():
    # TODO:         param.requires_grad = False
    if use_weights and model_name[-1] not in ['T', 'J']:
        weights = get_imagenet_weights(model_name=model_name)
        del weights['fully_connected.weight']
        del weights['fully_connected.bias']
        model.load_state_dict(weights, strict=False)


    return model


def train_chain_id(config: Dict, data_path: Path):
    print(config)
    ds_config = config['dataset']
    train_annotations = Path(data_path / ds_config['training'])
    train_ds = H5TripletHotelDataset(annotations_file_path=train_annotations, config=config['dataset'])
    val_annotations = Path(data_path / ds_config['validation'])
    val_ds = H5TripletHotelDataset(annotations_file_path=val_annotations, config=config['dataset'])

    checkpoint_dir = Path(repo_path / config['model_output'])

    trainer_config = TripletTrainer.Config.from_config(config['trainer'])

    trainer = TripletTrainer()
    
    class_net = get_model(config, num_classes=512)

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

    train_chain_id(config, data_path)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = ArgumentParser()
    parser.add_argument('config_path', type=Path, default='data/config/train_class_net.json')
    parser.add_argument('--data-path', type=Path, default=None)
    parser.add_argument('-m', '--model', type=str, default=None)
    main(parser.parse_args())