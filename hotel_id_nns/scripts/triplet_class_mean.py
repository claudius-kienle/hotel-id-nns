import torch
from torch.utils.data import DataLoader
import json
from pathlib import Path

from tqdm import tqdm
from hotel_id_nns.nn.modules.resnet_chris import ResNet, resnet50_cfg
from hotel_id_nns.nn.datasets.h5_hotel_dataset import H5HotelDataset
from hotel_id_nns.nn.datasets.h5_triplet_hotel_dataset import H5TripletHotelDataset
from hotel_id_nns.nn.modules.triplet_net import TripletNet
from hotel_id_nns.scripts.train_triplet_hotel_id import get_model

dir_path = Path(__file__).parent
repo_path = dir_path.parent.parent

def main():
    with open("data/configs/train_hotel_id_triplet.json", 'r') as f:
        config =json.load(f)

    config['model_name'] = 'ResNet50'


    dss = [
        "data/dataset/hotel_train_chain.h5",
        # "data/dataset/hotel_test_chain.h5",
        # "data/dataset/hotel_val_chain.h5"
    ]

    ds = H5HotelDataset(repo_path / dss[0], config=config['dataset'])
    dl = DataLoader(ds, batch_size=128)

    model = TripletNet(backbone=ResNet(resnet50_cfg, 128))
    # model.load_state_dict(torch.load(repo_path / "data/checkpoints/hotel-id-triplet/1673797952.4643068/e55.pth"))
    model_path = repo_path / "data/checkpoints/hotel-id-triplet/1673959543.7748818/e63.pth"
    model.load_state_dict(torch.load(model_path))
    model = model.to(torch.device('cuda'))

    features = [None] * ds.num_hotel_id_classes
    for idx in range(len(features)):
        features[idx] = []

    for batch in tqdm(dl, total=(len(ds) // dl.batch_size) + 1):
        imgs, _, hotel_ids = batch[:3]

        imgs = imgs.to(torch.device('cuda'))
        
        with torch.no_grad():
            feature_vector = model(imgs)

        for hotel_id, fv in zip(hotel_ids, feature_vector.cpu()):
            features[hotel_id.item()].append(fv)

    for idx, f in enumerate(features):
        if len(f) == 0:
            print("no samples for class %d" % idx)

    features = [
        torch.stack(feature_vec, dim=0) if len(feature_vec) > 0 else torch.full(size=(128,), fill_value=torch.nan)
        for feature_vec in features
    ]

    feature_means = torch.stack([
        torch.mean(feature, dim=0)
        for feature in features
    ], dim=0)

    print(feature_means.shape)

    torch.save(feature_means, model_path.parent / "means.pt")

    # out_dir = repo_path / "features"
    # out_dir.mkdir()
    # for idx, feature_mean in enumerate(feature_means):
    #     torch.save(feature_mean, out_dir / ("f_%s.pt" % idx))
    # feature_means = torch.stack(feature_means, dim=0)
    # print(feature_means.shape)

    # torch.save(feature_means, repo_path / 'features.pth')



if __name__ == "__main__":
    main()