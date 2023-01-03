from typing import Sized, Iterator, List
from torch.utils.data.sampler import RandomSampler


class TripletSampler(RandomSampler):
    def __init__(self, data_source: Sized):
        super().__init__(data_source)
        if not hasattr(data_source, "get_triplet_generator"):
            raise RuntimeError("Datasource must have triplet_generator function to support triplet sampling")
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        return self.data_source.get_triplet_generator()
