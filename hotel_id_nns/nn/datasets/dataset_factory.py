from pathlib import Path
from typing import Dict, Union
from torch.utils.data import Dataset
from hotel_id_nns.nn.datasets.hotel_dataset import HotelDataset
from hotel_id_nns.nn.datasets.chain_dataset_h5 import H5ChainDataset
from hotel_id_nns.nn.datasets.triplet_hotel_dataset import TripletHotelDataset

class DatasetFactory(object):

    def get(self, path: Path, config: Dict) -> Union[H5ChainDataset, HotelDataset, TripletHotelDataset]:
        actual_path = path

        # A default-switch which changes to the slower csv dataset instead of the faster h5 because
        # the annotation files for h5 dataset-loading are missing.
        if path.suffix == '.h5' and not path.exists():
            actual_path = path.with_suffix('.csv')
            print("File '" + str(path) + "' not found. Use .csv suffix instead")

        if actual_path.suffix == '.h5':
            return H5ChainDataset(annotations_file_path=actual_path, config=config)
        elif actual_path.suffix == '.csv':
            if "triplet_sampling" in config and config["triplet_sampling"]:
                return TripletHotelDataset(annotations_file_path=actual_path, config=config)
            else:
                return HotelDataset(annotations_file_path=actual_path, config=config)

        else:
            raise NotImplementedError()
