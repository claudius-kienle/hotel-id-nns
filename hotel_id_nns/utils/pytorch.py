from pathlib import Path
import torch


def load_model_weights(path: Path):
    weights = torch.load(path, map_location=torch.device('cpu'))
    if list(weights.keys())[0].startswith('module.'):
        weights_old = weights
        weights = {
            key.split("module.")[1]: weight
            for key, weight in weights_old.items()
        }
    return weights