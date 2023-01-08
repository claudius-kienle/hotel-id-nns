# This implementation is inspired by https://omoindrot.github.io/triplet-loss#triplet-mining
import torch


class TripletLoss(object):
    def __init__(self) -> None:
        self.loss = torch.nn.TripletMarginWithDistanceLoss(distance_function=TripletLoss.similarity)

    @staticmethod
    def similarity(a: torch.Tensor, b: torch.Tensor):
        # Returns a tensor of shape (batch) where each value is between 0 and 1,
        # 0 meaning a smaller distance and 1 meaning a maximum distance.
        return 1. - (torch.nn.functional.cosine_similarity(a, b) + 1.) / 2.

    def __call__(self, anchor_batch: torch.Tensor, pos_batch: torch.Tensor, neg_batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the batch triplet loss over axis 1.

        :param anchor_batch: shape=(batch, feature)
        :param pos_batch: shape=(batch, feature)
        :param neg_batch: shape=(batch, feature)
        :return: shape=(1)
        """
        return self.loss(anchor_batch, pos_batch, neg_batch)


if __name__ == "__main__":
    emb1 = torch.ones((4, 10))
    emb2 = -torch.ones((4, 10))
    emb3 = torch.ones((4, 10))

    loss = TripletLoss()

    print(TripletLoss.similarity(emb1, emb2))

    print(loss(emb1, emb2, emb3))


