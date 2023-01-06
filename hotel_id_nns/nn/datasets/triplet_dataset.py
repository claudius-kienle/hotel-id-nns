from pathlib import Path
from typing import Tuple

import torch
from hotel_id_nns.nn.datasets.h5_hotel_dataset import H5HotelDataset


class TripletDataset(H5HotelDataset):

    def __init__(self, annotations_file_path: Path, config: dict) -> None:
        super().__init__(annotations_file_path, config)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img, chain_id, hotel_id = super().__getitem__(index)

        # positive
        pos_anchors = (self.chain_ids == chain_id).nonzero()
        assert len(pos_anchors) > 1, 'atleast another positive sample'
        rand_index = index
        while rand_index == index:
            rand_index = pos_anchors[torch.randint(high=len(pos_anchors), size=(1,))]
        positive_anchor = self.preprocess(torch.as_tensor(self.imgs[rand_index]))
        
        # negative
        ## sample chain id, pick random sample with that chain id
        neg_chain_id = chain_id
        while neg_chain_id == chain_id:
            neg_chain_id = torch.randint(high=self.chain_ids.max(), size=(1,))
        neg_anchors = (self.chain_ids == neg_chain_id).nonzero()
        assert len(neg_anchors) >= 1, 'at least one negative sample'
        rand_index = neg_anchors[torch.randint(high=len(neg_anchors), size=(1,))]
        negative_anchor = self.preprocess(torch.as_tensor(self.imgs[rand_index]))

        return img, positive_anchor, negative_anchor


