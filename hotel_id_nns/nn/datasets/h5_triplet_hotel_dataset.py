import copy
from pathlib import Path
from typing import Tuple

import torch

from hotel_id_nns.nn.datasets.h5_hotel_dataset import H5HotelDataset


class H5TripletHotelDataset(H5HotelDataset):
    def __init__(self, class_name: str, annotations_file_path: Path, config: dict):
        super().__init__(annotations_file_path, config)

        # self.class_name = config["triplet_class_name"]
        self.class_name = class_name

        # Create class_to_indexes dictionary for the selected class_name
        if self.class_name == "chain_id":
            self._class_selector_idx = 0
        elif self.class_name == "hotel_id":
            self._class_selector_idx = 1
        else:
            raise RuntimeError(f"Unsupported class-name {self.class_name}")

        self.class_labels = self._select(self.chain_ids, self.hotel_ids)

    def _select(self, a, b):
        return [a, b][self._class_selector_idx]

    def __getitem__(self, a_ds_index: int) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                     torch.Tensor, torch.Tensor, torch.Tensor,
                     torch.Tensor, torch.Tensor, torch.Tensor]:
        class_labels = self.class_labels[self.valid_indices]

        a_img, a_chain_id, a_hotel_id = super().__getitem__(a_ds_index)
        a_label = class_labels[a_ds_index].item()

        # Calculate positive dataset index
        p_ds_indices = (class_labels == a_label).nonzero()
        p_ds_index = a_ds_index
        while p_ds_index == a_ds_index and len(p_ds_indices) > 1:
            p_ds_index = p_ds_indices[torch.randint(high=len(p_ds_indices), size=(1,))].item()
        p_img, p_chain_id, p_hotel_id = super().__getitem__(p_ds_index)

        # Calculate negative dataset index
        unique_classes = torch.unique(class_labels)
        unique_classes = unique_classes[unique_classes != a_label]
        n_label = unique_classes[torch.randint(high=len(unique_classes), size=(1,))]
        n_ds_indices = (class_labels == n_label).nonzero()
        n_ds_index = n_ds_indices[torch.randint(high=len(n_ds_indices), size=(1,))]

        # Always of a different class than a_img
        n_img, n_chain_id, n_hotel_id = super().__getitem__(n_ds_index.item())
        # print(a_ds_index, p_ds_index, n_ds_index)
        return a_img, p_img, n_img, a_chain_id, p_chain_id, n_chain_id, a_hotel_id, p_hotel_id, n_hotel_id




