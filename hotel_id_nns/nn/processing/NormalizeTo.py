from typing import Optional
import torch
from torchvision.transforms import functional as F


class NormalizeTo(torch.nn.Module):
    """Normalize a tensor image with mean and standard deviation.
    This transform does not support PIL Image.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(self, mean: Optional[torch.Tensor]=None, std: Optional[torch.Tensor]=None, inplace=False):
        super().__init__()

        self.mean = mean if mean is not None else torch.zeros(1)
        self.std = std if std is not None else torch.ones(1)
        self.inplace = inplace

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor (Tensor): Tensor image to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        dims = tuple(range(1,len(tensor)))
        mean = tensor.mean(dims)
        std = tensor.std(dims)
        out = F.normalize(tensor, mean, std, self.inplace) * self.std + self.mean
        return out

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
