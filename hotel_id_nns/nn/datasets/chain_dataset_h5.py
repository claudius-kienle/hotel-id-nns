from pathlib import Path
import time
from typing import Iterable, Tuple
import h5py
import PIL
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

        self.num_chain_id_classes = config['num_chain_id_classes']
        
        chain_id_weights_file = annotations_file_path.parent / "chain_id_weights.csv"
        assert chain_id_weights_file.exists()
        chain_id_weights = pd.read_csv(chain_id_weights_file, index_col='chain_id').sort_index()
        self.chain_id_weights = torch.as_tensor(chain_id_weights['weights'].values,  dtype=torch.float32)

        dataset = h5py.File(annotations_file_path)

        self.imgs = dataset['img']
        self.chain_ids = dataset['chain_id']

        self.preprocess = T.Compose([
            T.ConvertImageDtype(dtype=torch.float32),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    def __len__(self) -> int:
        return len(self.imgs)

    def __iter__(self) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        for img, chain_id in zip(self.imgs, self.chain_ids):
            img = self.preprocess(torch.as_tensor(img))
            yield img, torch.as_tensor(chain_id, dtype=torch.long)

    def __getitem__(self, index) -> torch.Tensor:
        img = torch.as_tensor(self.imgs[index])
        chain_id = torch.as_tensor(self.chain_ids[index], dtype=torch.long)

        img = self.preprocess(img)

        return img, chain_id
