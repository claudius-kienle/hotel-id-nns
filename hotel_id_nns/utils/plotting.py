from matplotlib.axes import Axes
import torch
from matplotlib import pyplot as plt


def plot_confusion_matrix(cm: torch.Tensor, ax: Axes):
    ax.matshow(cm)
    ax.set_xlabel('Predicted Chain ID')
    ax.set_ylabel('True Chain ID')
