import torch
from matplotlib import pyplot as plt


def plot_confusion_matrix(cm: torch.Tensor):
    plt.matshow(cm)
    plt.xlabel('Predicted Chain ID')
    plt.ylabel('True Chain ID')
    plt.show()
