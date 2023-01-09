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


def to_hdf5(ds_path, parallel=True):
    ds = pd.read_csv(ds_path)
    img_dir = ds_path.parent / "train_images"
    hdf5_path = ds_path.parent / ("%s.h5" % ds_path.stem)

    if hdf5_path.exists():
        hdf5_path.unlink()

    img_size = 256
    crop_size = 224

    preprocess = T.Compose([
        T.Resize(size=(img_size, img_size)),
        T.CenterCrop(size=(crop_size, crop_size)),
        T.PILToTensor(),
    ])

    with h5py.File(hdf5_path, 'w') as f:
        train_ds_imgs = f.create_dataset('img', (len(ds.index), 3, crop_size, crop_size),
                                         dtype=np.uint8)
        train_ds_chain_ids = f.create_dataset('chain_id', (len(ds.index), ), dtype=np.int32)
        train_ds_hotel_ids = f.create_dataset('hotel_id', (len(ds.index), ), dtype=np.int32)

        print('creating tensors...')

        def process(idx, train_img):
            chain_id = train_img.chain_id
            hotel_id = train_img.hotel_id
            img_path = train_img.path

            img = preprocess(Image.open(img_dir / img_path))

            return img, chain_id, hotel_id

        if parallel:
            imgs_chain_ids = Parallel(n_jobs=-1)(
                delayed(process)(idx, train_img)
                for idx, train_img in tqdm(ds.iterrows(), desc='Conversion', total=len(ds.index)))

            imgs, chain_ids, hotel_ids = zip(*imgs_chain_ids)

            train_ds_imgs[:] = torch.stack(imgs, dim=0).numpy()
            train_ds_chain_ids[:] = torch.as_tensor(chain_ids).numpy()
            train_ds_hotel_ids[:] = torch.as_tensor(hotel_ids).numpy()

        else:
            for idx, train_img in tqdm(ds.iterrows(), desc='Conversion', total=len(ds.index)):
                img, chain_id, hotel_id = process(idx, train_img)
                train_ds_imgs[idx] = img.numpy()
                train_ds_chain_ids[idx] = torch.as_tensor(chain_id).numpy()
                train_ds_hotel_ids[idx] = torch.as_tensor(hotel_id).numpy()


def main():
    train_ds_path = root_dir / "data/dataset/hotel_train_chain.csv"
    test_ds_path = root_dir / "data/dataset/hotel_test_chain.csv"
    val_ds_path = root_dir / "data/dataset/hotel_val_chain.csv"

    parallel = True

    to_hdf5(test_ds_path, parallel)
    to_hdf5(train_ds_path, parallel)
    to_hdf5(val_ds_path, parallel)


if __name__ == "__main__":
    main()
