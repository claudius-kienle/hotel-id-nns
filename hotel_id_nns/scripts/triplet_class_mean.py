import torch
from torch.utils.data import DataLoader
import json
from pathlib import Path

from tqdm import tqdm
from hotel_id_nns.nn.datasets.h5_hotel_dataset import H5HotelDataset
from hotel_id_nns.nn.datasets.h5_triplet_hotel_dataset import H5TripletHotelDataset
from hotel_id_nns.scripts.train_triplet_hotel_id import get_model

dir_path = Path(__file__).parent
repo_path = dir_path.parent.parent

def means(ds_path: str, model, config):
    ds = H5HotelDataset(repo_path / ds_path, config=config['dataset'])
    dl = DataLoader(ds, batch_size=16)

    features = [None] * ds.num_hotel_id_classes
    for idx in range(len(features)):
        features[idx] = []

    for batch in tqdm(dl, total=(len(ds) // dl.batch_size) + 1):
        imgs, _, hotel_ids = batch[:3]
        
        with torch.no_grad():
            feature_vector = model(imgs)

        for hotel_id, fv in zip(hotel_ids, feature_vector):
            features[hotel_id.item()].append(fv)

    return features

def main():
    with open("data/configs/train_hotel_id_triplet.json", 'r') as f:
        config =json.load(f)

    config['model_name'] = 'ResNet50'

    model = get_model(config, num_classes=512)

    dss = [
        "data/dataset/hotel_test_chain.h5",
        # "data/dataset/hotel_train_chain.h5",
        # "data/dataset/hotel_val_chain.h5"
    ]

    features = means(dss[0], model, config)

    feature_means = torch.concat([
        torch.mean(torch.stack(feature_vec, dim=0))
        for feature_vec in features
    ])

    torch.save(feature_means, repo_path / 'features.pth')



if __name__ == "__main__":
    main()