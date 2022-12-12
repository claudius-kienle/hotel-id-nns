from functools import reduce
from pathlib import Path
from typing import Dict, List
import torch
from torch import nn
from torch import optim


def load_model_weights(path: Path):
    weights = torch.load(path, map_location=torch.device('cpu'))
    if list(weights.keys())[0].startswith('module.'):
        weights_old = weights
        weights = {key.split("module.")[1]: weight for key, weight in weights_old.items()}
    return weights


def compute_metrics(pred_chain_id_probs: torch.Tensor, chain_ids: torch.Tensor):
    num_classes = pred_chain_id_probs.shape[-1]
    pred_chain_ids = torch.argmax(pred_chain_id_probs, dim=-1)

    indices = num_classes * chain_ids + pred_chain_ids
    cm = torch.bincount(indices, minlength=num_classes**2)\
        .reshape((num_classes, num_classes))

    accuracy = cm.diag().sum() / (cm.sum() + 1e-15)
    precision = cm.diag() / (cm.sum(dim=0) + 1e-15)
    recall = cm.diag() / (cm.sum(dim=1) + 1e-15)
    f1 = 2 * precision * recall / (precision + recall + 1e-15)
    unique_chain_ids = len(torch.unique(chain_ids))

    return {
        "accuracy": accuracy,
        "precision": precision.sum() / unique_chain_ids,
        "recall": recall.sum() / unique_chain_ids,
        "f1": f1.sum() / unique_chain_ids,
        "cm": cm,
    }

def aggregate_metics(metricsb: List[Dict]) -> Dict:
    metrics = {key: [metrics[key] for metrics in metricsb] for key in metricsb[0].keys()}
    reduced_metrics = {}
    for key, values in metrics.items():
        values = torch.hstack(values)
        if key  == 'cm':
            reduced_metrics[key] = values.sum(dim=0)
        else:
            reduced_metrics[key] = values.mean()
    return reduced_metrics


def get_optimizer(net: nn.Module, name: str, weight_decay: float,
                  learning_rate: float) -> optim.Optimizer:
    if name == 'rmsprop':
        return optim.RMSprop(net.parameters(),
                             lr=learning_rate,
                             weight_decay=weight_decay,
                             momentum=0.9)
    elif name == 'adam':
        return optim.Adam(net.parameters(),
                          lr=learning_rate,
                          betas=(0.9, 0.999),
                          eps=1e-08,
                          weight_decay=weight_decay,
                          amsgrad=False)
    elif name == 'amsgrad':
        return optim.Adam(net.parameters(),
                          lr=learning_rate,
                          betas=(0.9, 0.999),
                          eps=1e-08,
                          weight_decay=weight_decay,
                          amsgrad=False)
    elif name == 'sgd':
        return optim.SGD(
            net.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        raise RuntimeError(f"invalid optimizer name given {name}")