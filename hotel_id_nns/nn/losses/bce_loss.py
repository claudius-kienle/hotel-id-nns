import torch


class BCELoss(object):


    def __init__(self) -> None:
        pass

    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        w, h = input.shape[-2:]
        pixels = w * h
        return torch.nn.functional.binary_cross_entropy(input=input, target=input) * pixels