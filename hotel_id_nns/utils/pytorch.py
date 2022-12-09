from pathlib import Path
import torch


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
