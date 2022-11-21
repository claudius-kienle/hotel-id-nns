from pathlib import Path
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from hotel_id_nns.nn.processing.NormalizeTo import NormalizeTo

class ChainDataset(Dataset):

    def __init__(self, annotations_file_path: Path, size: int) -> None:
        super().__init__()
        assert annotations_file_path.exists()

        self.ds_path = annotations_file_path.parent / "train_images"
        self.chain_annotations = pd.read_csv(annotations_file_path, names=['path', 'chain_id'], sep=' ')

        self.preprocess = T.Compose([
            T.RandomCrop(size=size),
            T.PILToTensor(),
            T.ConvertImageDtype(dtype=torch.float32),
            NormalizeTo(),
        ])


    def __len__(self) -> int:
        return len(self.chain_annotations)

    def __getitem__(self, index) -> torch.Tensor:
        entry = self.chain_annotations.iloc[index]
        file_path = self.ds_path / entry.path

        img = Image.open(file_path)

        img_tensor = self.preprocess(img).to(torch.float32)
        return img_tensor, entry.chain_id

