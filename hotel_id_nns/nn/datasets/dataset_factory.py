from pathlib import Path
from typing import Dict, Union
from torch.utils.data import Dataset
from hotel_id_nns.nn.datasets.chain_dataset import ChainDataset
from hotel_id_nns.nn.datasets.chain_dataset_h5 import H5ChainDataset


class DatasetFactory(object):

    def get(self, path: Path, config: Dict) -> Union[H5ChainDataset, ChainDataset]:
        if path.suffix == '.h5':
            return H5ChainDataset(annotations_file_path=path, config=config)
        elif path.suffix == '.csv':
            return ChainDataset(annotations_file_path=path, config=config)
        else:
            raise NotImplementedError()