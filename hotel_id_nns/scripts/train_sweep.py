
from argparse import ArgumentParser
import json
import logging
import os
from pathlib import Path

import wandb

from hotel_id_nns.scripts.train_classification import train_chain_id


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

    wandb.init(project="hotel-id", resume='allow', entity="hotel-id-nns", reinit=True)

    # update config with wandb
    tc = config['trainer']
    # tc['optimizer_name'] =  wandb.config.optimizer_name
    tc['weight_decay'] = wandb.config.weight_decay
    config['model_name'] = "ResNet50"
    # tc['lr_patience'] = wandb.config.lr_patience

    train_chain_id(config, data_path)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = ArgumentParser()
    parser.add_argument('config_path', type=Path)
    main(parser.parse_args())