import argparse
from pathlib import Path
from enum import Enum
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from hotel_id_nns.nn.datasets.h5_hotel_dataset import H5HotelDataset
from hotel_id_nns.nn.datasets.h5_triplet_hotel_dataset import H5TripletHotelDataset
from hotel_id_nns.nn.modules.resnet_chris import ResNet, resnet50_cfg
from hotel_id_nns.nn.modules.triplet_net import TripletNet
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
    ds = H5HotelDataset(annotations_file_path=root_dir / args.dataset_path, config=config)
    print(len(ds))

    # features = torch.load(root_dir / "means.pt")
    features = torch.load((root_dir / args.model_path).parent / "means.pt")

    if class_type == ClassType.chain_id:
        num_classes = ds.num_chain_id_classes
    elif class_type == ClassType.hotel_id:
        num_classes = ds.num_hotel_id_classes

    class_net = TripletNet(backbone=ResNet(network_cfg=resnet50_cfg, out_features=128))
    class_net.load_state_dict(load_model_weights(root_dir / args.model_path))
    class_net = class_net.to('cuda')
    class_net.eval()

    # load model weights and map if was DataParallel model

    mapping = torch.load((root_dir / args.dataset_path).parent / "hotel-chain-id.pt")
    ds = DataLoader(ds, batch_size=args.batch_size)
    mapping = torch.load((root_dir / args.dataset_path).parent / "hotel-chain-id.pt")
    
    # generate predictions on dataset
    gt = []
    preds = []

    metricsb = []
    cm = None
    idx = 0
    ds = tqdm(ds)
    with torch.no_grad():
        for sample in ds:
            input_img, chain_id, hotel_id = sample
            # input_img, p_img, n_img, chain_id, _, _, hotel_id, _, _ = sample

            if class_type == ClassType.chain_id:
                label = chain_id
            elif class_type == ClassType.hotel_id:
                label = hotel_id

            input_img = input_img.to("cuda")

            pred_features = class_net(input_img).cpu()
            # pos_pred_features = class_net(p_img)
            # neg_pred_features = class_net(n_img)
            if True:
                pred_label_probs = torch.nn.CosineSimilarity(dim=-1)(pred_features[:, None], features[None])
                pred_label_probs = pred_label_probs / torch.sum(pred_label_probs, dim=1, keepdim=True)
            else:
                pred_label_probs = torch.sqrt(torch.sum((pred_features[:, None] - features[None]) ** 2, dim=-1))
                pred_label_probs = 1 - pred_label_probs / torch.sum(pred_label_probs, dim=1, keepdim=True)

            if class_type == ClassType.chain_id:
                valid_hotel_ids = mapping != -1
                mapping[~valid_hotel_ids] = 0 
                mapping_onehot = torch.nn.functional.one_hot(mapping)
                mapping_onehot[valid_hotel_ids] = 0

                # pred_label_probs = pred_label_probs @ mapping_onehot.to(torch.float32)
                pred_label_probs = (pred_label_probs[:, :, None] * mapping_onehot).max(dim=1)[0]

            pred_label_probs = pred_label_probs / torch.sum(pred_label_probs, dim=0, keepdim=True)
            _, pred_label = torch.max(pred_label_probs, dim=1)
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
            if 'train' in args.dataset_path.as_posix(): # stop early on train datset
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
    parser.add_argument("-d", "--dataset-path",type=Path, default="data/dataset/hotel_test_chain.h5")
    parser.add_argument('--batch-size', type=int, default=128)
    main(parser.parse_args())