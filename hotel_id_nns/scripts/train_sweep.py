
from argparse import ArgumentParser
import json
import logging
import os
from pathlib import Path

import wandb

from hotel_id_nns.scripts.train_chain_id import train_chain_id


dir_path = Path(__file__).parent
repo_path = dir_path.parent.parent

def main(args):
    config_file = repo_path / args.config_path
    assert config_file.suffix == '.json'

    with (config_file).open(mode="r") as f:
        config = json.load(f)

    # check for bwunicluster dataset
    tmp_dir = Path(os.environ['TMP']) 
    data_path = tmp_dir if (tmp_dir / "data/dataset").exists() else repo_path

    wandb.init(project="ClassNet", resume='allow', entity="hotel-id-nns", reinit=True)

    # update config with wandb
    tc = config['trainer']
    # tc['optimizer_name'] =  wandb.config.optimizer_name
    tc['learning_rate'] = wandb.config.learning_rate
    tc['weight_decay'] = wandb.config.weight_decay
    # tc['lr_patience'] = wandb.config.lr_patience
    config['trainer'] = tc
    config['model_name'] = wandb.config.model_name
    config['model_weights_imagenet'] = wandb.config.model_weights_imagenet
    config['model_finetune'] = wandb.config.model_finetune

    train_chain_id(config, data_path)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = ArgumentParser()
    parser.add_argument('config_path', type=Path, default='data/config/train_class_net.json')
    main(parser.parse_args())