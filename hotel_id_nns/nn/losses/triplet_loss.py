# This implementation is inspired by https://omoindrot.github.io/triplet-loss#triplet-mining
import torch


def _compute_pairwise_distance_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    embedding_dot_products = embeddings @ embeddings.T
    # embedding_dot_products[i, j] == torch.dot(embeddings[i], embeddings[j])

    # embedding_norms is a column vector (norm for each embedding)
    embedding_norms = torch.norm(embeddings, dim=1).reshape(embeddings.shape[0], 1)

    # norm_multiplications[i, j] = embedding_norms[i] * embedding_norms[j]
    norm_multiplications = embedding_norms @ embedding_norms.T

    return torch.div(embedding_dot_products, norm_multiplications)  # cos(phi)


def _batch_triplet_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 0.) -> torch.Tensor:
    pairwise_distances = _compute_pairwise_distance_matrix(embeddings)

    print(pairwise_distances)

    anchor_pos_dist = torch.unsqueeze(pairwise_distances, 2)
    anchor_neg_dist = torch.unsqueeze(pairwise_distances, 1)

    triplet_loss = anchor_pos_dist - anchor_neg_dist + margin

    # a, p, n -> (labels[a] != labels[p]) or (labels[n] == labels[a]) or (a == p)
    print(triplet_loss)


class TripletLoss(object):
    def __init__(self) -> None:
        pass

    def __call__(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pass


if __name__ == "__main__":
    emb = torch.tensor([[1., 1., -1.],
                        [-2., 0.5, 1.]])
    labels = torch.tensor([0, 1, 2])


    # _batch_triplet_loss(emb, None)

