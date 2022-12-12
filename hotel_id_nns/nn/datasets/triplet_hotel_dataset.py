import random
from typing import Tuple

from torch import Tensor
import copy

from hotel_id_nns.nn.datasets.hotel_dataset import HotelDataset
import torch
from pathlib import Path


class TripletHotelDataset(HotelDataset):
    def __init__(self, annotations_file_path: Path, config: dict):
        super().__init__(annotations_file_path=annotations_file_path, config=config)

        self.random = random.Random(0)

        # generate sample dict structured by label
        self.samples_per_class = dict()
        for idx, row in self.class_annotations.iterrows():
            label = row[self._class_name]

            # print(label, idx, row["path"])
            if label not in self.samples_per_class:
                self.samples_per_class[label] = []

            self.samples_per_class[label].append(idx)

    def __getitem__(self, index) -> torch.Tensor:
        return super().__getitem__(index)

    def get_triplet_generator(self):
        even = True
        idx = None
        while True:
            if even:
                even = False
                idx = self.random.randrange(0, len(self))
            else:
                even = True
                # calc idx from same class
                label = self.class_annotations.iloc[idx][self._class_name]
                samples_with_label = self.samples_per_class[label]
                samples_with_label.remove(idx)

                if len(samples_with_label) == 0:
                    continue

                idx = random.choice(samples_with_label)

            yield idx
