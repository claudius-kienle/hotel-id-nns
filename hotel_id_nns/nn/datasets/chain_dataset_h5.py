from pathlib import Path
from re import S
import time
from typing import Iterable, Tuple
import h5py
import PIL
from numpy import indices
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, IterableDataset

from hotel_id_nns.nn.processing.NormalizeTo import NormalizeTo

PIL.Image.MAX_IMAGE_PIXELS = 933120000

class H5ChainDataset(Dataset):

    def __init__(self, annotations_file_path: Path, config: dict) -> None:
        # TODO: adapt size to tuple (w,h)
        super().__init__()
        print("loading dataset from path %s" % annotations_file_path.as_posix())

        assert annotations_file_path.exists()

        self.num_chain_id_classes = config['num_chain_id_classes'] - 1
        
        chain_id_weights_file = annotations_file_path.parent / "chain_id_weights.csv"
        assert chain_id_weights_file.exists()
        # re-weight, as chain_id 0 will be ignored (stands for 'chain_id unknown')
        chain_id_weights = pd.read_csv(chain_id_weights_file, index_col='chain_id').sort_index()
        chain_id_weights.weights = chain_id_weights.weights / chain_id_weights[1:].weights.sum()
        chain_id_weights = chain_id_weights[1:]

        self.chain_id_weights = torch.as_tensor(chain_id_weights['weights'].values,  dtype=torch.float32)

        dataset = h5py.File(annotations_file_path)

        # only iterate over samples who's chain_id is not zero. Therefore compute indices of whole dataset where this is the case
        self.imgs = dataset['img']
        self.chain_ids = torch.as_tensor(dataset['chain_id'], dtype=torch.long).squeeze()

        indices_for_each_class = []
        for i in range(1, self.num_chain_id_classes):
            indices_of_chain_ids = torch.nonzero(self.chain_ids == i).squeeze()
            if indices_of_chain_ids.numel() > 1:
                indices_for_each_class.append(indices_of_chain_ids[0])
            elif indices_of_chain_ids.numel() == 1:
                indices_for_each_class.append(indices_of_chain_ids)
        self.chain_id_non_zero = torch.vstack(indices_for_each_class)
        self.chain_id_weights[:] = 1 / self.num_chain_id_classes
        torch.set_printoptions(edgeitems=100)
        print(self.chain_id_non_zero.squeeze())
        print(self.chain_ids[self.chain_id_non_zero].squeeze())

        # self.chain_id_non_zero = torch.nonzero(self.chain_ids)

        self.preprocess = T.Compose([
            T.ConvertImageDtype(dtype=torch.float32),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self) -> int:
        return len(self.chain_id_non_zero)

    # def __iter__(self) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    #     for img, chain_id in zip(self.imgs, self.chain_ids):
    #         if chain_id == 0:
    #             continue
    #         img = self.preprocess(torch.as_tensor(img))
    #         yield img, torch.as_tensor(chain_id, dtype=torch.long)

    def __getitem__(self, index) -> torch.Tensor:
        idx = self.chain_id_non_zero[index]
        img = torch.as_tensor(self.imgs[idx])
        chain_id = self.chain_ids[idx] - 1

        # chains = self.chain_ids[self.chain_id_non_zero]

        img = self.preprocess(img)

        return img, chain_id
