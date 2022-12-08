import torch


class TripletLoss(object):
    def __init__(self, margin: float = 1.) -> None:
        self.margin = margin

    def __call__(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """
        N is the batch-dimension
        F is the single feature-dimension

        If The training data consists of images, the single feature-dimension can be achieved by
        e.g. global average pooling.

        :param anchor: shape=(N, F)
        :param positive: shape=(N, F)
        :param negative: shape=(N, F)
        :return: shape=(1)
        """

        assert len(anchor.shape) == 2
        return torch.nn.functional.triplet_margin_loss(anchor, positive, negative, margin=self.margin)
