from pathlib import Path
import pandas as pd
import h5py
from joblib import Parallel, delayed
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

root_dir = Path(__file__).parent.parent.parent

def to_hdf5(ds_path):
    ds = pd.read_csv(ds_path, names=['path', 'chain_id'], sep=' ')
    img_dir = ds_path.parent / "train_images"
    hdf5_path = ds_path.parent / ("%s.h5" % ds_path.stem)

    img_size = 256
    crop_size = 224

    preprocess = T.Compose([
        T.Resize(size=(img_size, img_size)),
        T.CenterCrop(size=(crop_size, crop_size)),
        T.PILToTensor(),
    ])

    with h5py.File(hdf5_path, 'w') as f:
        train_ds_imgs = f.create_dataset('img', (len(ds.index), 3, crop_size, crop_size), dtype=np.float32)
        train_ds_chain_ids = f.create_dataset('chain_id', (len(ds.index),), dtype=np.int32)

        imgs = torch.empty((len(ds.index), 3, crop_size, crop_size))
        chain_ids = torch.empty((len(ds.index),))

        for idx, train_img in tqdm(ds.iterrows(), desc='Conversion', total=len(ds.index)):
            chain_id = train_img.chain_id
            img_path = train_img.path

            img = preprocess(Image.open(img_dir / img_path))

            train_ds_imgs[idx] = img
            train_ds_chain_ids[idx] = chain_id


def main():
    train_ds_path = root_dir / "data/dataset/hotel_train_chain.csv"
    test_ds_path = root_dir / "data/dataset/hotel_test_chain.csv"
    val_ds_path = root_dir / "data/dataset/hotel_val_chain.csv"
    to_hdf5(train_ds_path)
    to_hdf5(test_ds_path)
    to_hdf5(val_ds_path)

if __name__ == "__main__":
    main()