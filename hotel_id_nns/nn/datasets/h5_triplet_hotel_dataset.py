import copy
from pathlib import Path
from typing import Tuple

import torch

from hotel_id_nns.nn.datasets.h5_hotel_dataset import H5HotelDataset


class H5TripletHotelDataset(H5HotelDataset):
    def __init__(self, annotations_file_path: Path, config: dict):
        super().__init__(annotations_file_path, config)

        # self.class_name = config["triplet_class_name"]
        self.class_name = "hotel_id"

        # Create class_to_indexes dictionary for the selected class_name
        if self.class_name == "chain_id":
            self._class_selector_idx = 0
        elif self.class_name == "hotel_id":
            self._class_selector_idx = 1
        else:
            raise RuntimeError(f"Unsupported class-name {self.class_name}")

        self.class_labels = self._select(self.chain_ids, self.hotel_ids)

        label_to_indexes = dict()
        for dataset_idx, idx in enumerate(self.valid_indices):
            label = self.class_labels[idx].item()

            if label not in label_to_indexes:
                label_to_indexes[label] = []

            label_to_indexes[label].append(dataset_idx)

        self.label_to_indexes = label_to_indexes

    def _select(self, a, b):
        return [a, b][self._class_selector_idx]

    def __getitem__(self, a_ds_index: int) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                     torch.Tensor, torch.Tensor, torch.Tensor,
                     torch.Tensor, torch.Tensor, torch.Tensor]:
        a_img, a_chain_id, a_hotel_id = super().__getitem__(a_ds_index)
        a_label = self.class_labels[self.valid_indices[a_ds_index]].item()

        # Calculate positive dataset index
        p_ds_indexes: list = copy.deepcopy(self.label_to_indexes[a_label])

        if len(p_ds_indexes) == 1:
            p_ds_index = a_ds_index
        else:
            p_ds_indexes.remove(a_ds_index)
            p_ds_index = p_ds_indexes[torch.randint(high=len(p_ds_indexes), size=(1,)).item()]

        # Might be the same as a_img, ... because there is just one
        # sample for that class.
        p_img, p_chain_id, p_hotel_id = super().__getitem__(p_ds_index)

        # Calculate negative dataset index
        assert len(self.label_to_indexes) > 0  # more than one class
        n_ds_index = a_ds_index
        while self.class_labels[self.valid_indices[n_ds_index]].item() == a_label:
            n_ds_index = torch.randint(high=len(self.valid_indices), size=(1,)).item()

        # Always of a different class than a_img
        n_img, n_chain_id, n_hotel_id = super().__getitem__(n_ds_index)

        return a_img, p_img, n_img, a_chain_id, p_chain_id, n_chain_id, a_hotel_id, p_hotel_id, n_hotel_id




