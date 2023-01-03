from pathlib import Path
from typing import Tuple
import h5py
import PIL
import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset


PIL.Image.MAX_IMAGE_PIXELS = 933120000

class H5HotelDataset(Dataset):

    @staticmethod
    def __load_weights(path) -> torch.Tensor:
        assert path.exists()
        # re-weight, as chain_id 0 will be ignored (stands for 'chain_id unknown')
        weights_df = pd.read_csv(path, index_col=0).sort_index()
        weights_df.weights = weights_df.weights / weights_df[1:].weights.sum() # ensure normalized
        return torch.as_tensor(weights_df['weights'].values, dtype=torch.float32)

    def __init__(self, annotations_file_path: Path, config: dict) -> None:
        # TODO: adapt size to tuple (w,h)
        super().__init__()
        print("loading dataset from path %s" % annotations_file_path.as_posix())

        assert annotations_file_path.exists()

        chain_id_weights_file = annotations_file_path.parent / "chain_id_weights.csv"
        self.chain_id_weights = self.__load_weights(chain_id_weights_file)
        self.num_chain_id_classes = len(self.chain_id_weights)

        hotel_id_weights_file = annotations_file_path.parent / 'hotel_id_weights.csv'
        self.hotel_id_weights = self.__load_weights(hotel_id_weights_file)
        self.num_hotel_id_classes = len(self.hotel_id_weights)

        dataset = h5py.File(annotations_file_path)

        self.imgs = dataset['img']
        self.chain_ids = torch.as_tensor(dataset['chain_id'], dtype=torch.long).squeeze()
        self.hotel_ids = torch.as_tensor(dataset['hotel_id'], dtype=torch.long).squeeze()

        if config['remove_unkown_chain_id_samples']:
            # only iterate over samples who's chain_id is not zero. Therefore compute indices of such valid samples
            self.valid_indices = torch.nonzero(self.chain_ids)
            # reduce chain_ids number by 1 since chain_id=0 removed
            self.chain_ids = self.chain_ids - 1
            self.chain_id_weights = self.chain_id_weights[1:]
            self.num_chain_id_classes = self.num_chain_id_classes - 1
        else:
            self.valid_indices = torch.arange(len(self.chain_ids))

        self.preprocess = T.Compose([
            T.ConvertImageDtype(dtype=torch.float32),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = self.valid_indices[index]
        img = torch.as_tensor(self.imgs[idx])
        chain_id = self.chain_ids[idx] 
        hotel_id = self.hotel_ids[idx] 

        img = self.preprocess(img)

        return img, chain_id, hotel_id
