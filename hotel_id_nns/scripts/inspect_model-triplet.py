import argparse
from pathlib import Path
from enum import Enum
import copy
from matplotlib import pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
from hotel_id_nns.nn.datasets.h5_hotel_dataset import H5HotelDataset
from hotel_id_nns.nn.datasets.h5_triplet_hotel_dataset import H5TripletHotelDataset
from hotel_id_nns.nn.modules.resnet_chris import ResNet, resnet50_cfg
from hotel_id_nns.utils.plotting import plot_confusion_matrix
from hotel_id_nns.utils.pytorch import compute_classification_metrics, load_model_weights

root_dir = Path(__file__).parent.parent.parent

class ClassType(Enum):
    hotel_id = 0
    chain_id = 1

def main(args):
    class_type = ClassType.hotel_id
    config = {
        "remove_unkown_chain_id_samples": class_type == ClassType.chain_id
    }
    ds = H5TripletHotelDataset(annotations_file_path=root_dir / args.dataset_path, config=config)

    features = torch.load(root_dir / "features.pth")

    if class_type == ClassType.chain_id:
        num_classes = ds.num_chain_id_classes
    elif class_type == ClassType.hotel_id:
        num_classes = ds.num_hotel_id_classes

    class_net =  ResNet(network_cfg=resnet50_cfg, out_features=512)
    class_net.load_state_dict(load_model_weights(root_dir / args.model_path))

    # load model weights and map if was DataParallel model

    ds = DataLoader(ds, batch_size=args.batch_size)
    
    # generate predictions on dataset
    gt = []
    preds = []

    metricsb = []
    cm = None
    idx = 0
    ds = tqdm(ds)
    for sample in ds:
        # input_img, chain_id, hotel_id = sample
        input_img, p_img, n_img, chain_id, _, _, hotel_id, _, _ = sample

        if class_type == ClassType.chain_id:
            label = chain_id
        elif class_type == ClassType.hotel_id:
            label = hotel_id

        pred_features = class_net(input_img)
        pos_pred_features = class_net(p_img)
        neg_pred_features = class_net(n_img)

        pred_label_probs = torch.sqrt(torch.sum((pred_features[:, None] - features[None]) ** 2, dim=-1))

        from hotel_id_nns.nn.losses.triplet_loss import TripletLoss
        # TripletLoss()(pred_features, features[hotel_id], None)
        TripletLoss()(pred_features, pos_pred_features, neg_pred_features)

        _, pred_label = torch.min(pred_label_probs, dim=1)
        pred_label_probs = 1 - pred_label_probs / torch.sum(pred_label_probs, dim=1, keepdim=True)
        gt.append(label)
        preds.append(pred_label)
        metrics = compute_classification_metrics(pred_label_probs, label.squeeze())

        if 'cm' in metrics:
            if cm is None:
                cm = metrics['cm']
            else:
                cm = cm + metrics['cm']
            metrics.pop('cm')
        metricsb.append(metrics)

        prints = {key: value.item() for key, value in metrics.items()}
        ds.set_postfix(prints)
        idx += 1
        if idx == 40:
            break

    gt = torch.concat(gt).squeeze()
    preds = torch.concat(preds)

    metrics = {key: [metrics[key] for metrics in metricsb] for key in metricsb[0].keys()}
    metrics = {key: torch.mean(torch.vstack(metrics)).item() for key, metrics in metrics.items()}
    print(metrics)

    ax = plt.subplot()
    plot_confusion_matrix(cm, ax)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("-d", "--dataset-path",type=Path, default="data/dataset/hotel_train_chain.h5")
    parser.add_argument('--batch-size', type=int, default=32)
    main(parser.parse_args())