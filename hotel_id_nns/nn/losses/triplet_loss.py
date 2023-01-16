import torch
import torch.nn.functional as F

class TripletLoss():

    def __init__(self, distance_function, margin: float = 0.2) -> None:
        self.margin = margin 
        self.distance_function = distance_function

    def __call__(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor):
        positive_dist = self.distance_function(anchor, positive)
        negative_dist = self.distance_function(anchor, negative)

        output = torch.clamp(positive_dist - negative_dist + self.margin, min=0.0)

        info = {
            "negative": negative_dist.mean(),
            "positive": positive_dist.mean()
        }

        return output.mean(), info
