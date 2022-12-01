from pathlib import Path
import time
import PIL
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from hotel_id_nns.nn.processing.NormalizeTo import NormalizeTo

PIL.Image.MAX_IMAGE_PIXELS = 933120000

class ChainDataset(Dataset):

    def __init__(self, annotations_file_path: Path, config: dict) -> None:
        # TODO: adapt size to tuple (w,h)
        super().__init__()
        print("loading dataset from path %s" % annotations_file_path.as_posix())

        assert annotations_file_path.exists()

        size = config['input_size']

        self.num_chain_id_classes = config['num_chain_id_classes']

        self.ds_path = annotations_file_path.parent / "train_images"
        self.chain_annotations = pd.read_csv(annotations_file_path, names=['path', 'chain_id'], sep=' ')

        preprocess_steps = []
        if size is not None:
            preprocess_steps.append( T.Resize(size=(size, size)))
        preprocess_steps.extend([
            T.PILToTensor(),
            T.ConvertImageDtype(dtype=torch.float32),
            # NormalizeTo(),
        ])

        self.preprocess = T.Compose(preprocess_steps)


    def __len__(self) -> int:
        return len(self.chain_annotations)

    def __getitem__(self, index) -> torch.Tensor:
        # t1 = time.time()
        entry = self.chain_annotations.iloc[index]
        file_path = self.ds_path / entry.path

        img = Image.open(file_path)
        img_tensor = self.preprocess(img).to(torch.float32)

        chain_id = torch.as_tensor(entry.chain_id)
        # onehot_chain_id = torch.nn.functional.one_hot(chain_id, num_classes=self.num_chain_id_classes)
        # print("img loading took %e" % (time.time() - t1))
        return img_tensor, chain_id

