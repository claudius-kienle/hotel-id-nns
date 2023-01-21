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
from hotel_id_nns.nn.modules.resnet_chris import ResNet, resnet50_cfg
from hotel_id_nns.utils.plotting import plot_confusion_matrix
from hotel_id_nns.nn.modules.tripet_classification_net import TripletClassificationNet
from hotel_id_nns.nn.modules.triplet_net import TripletNet
from hotel_id_nns.utils.pytorch import compute_classification_metrics, load_model_weights

root_dir = Path(__file__).parent.parent.parent

class ClassType(Enum):
    hotel_id = 0
    chain_id = 1

def main(args):
    config = {
        "remove_unkown_chain_id_samples": True
    }
    class_type = ClassType.hotel_id
    ds = H5HotelDataset(annotations_file_path=root_dir / args.dataset_path, config=config)

    if class_type == ClassType.chain_id:
        num_classes = ds.num_chain_id_classes
    elif class_type == ClassType.hotel_id:
        num_classes = ds.num_hotel_id_classes
    
    weights = load_model_weights(root_dir / args.model_path)

    if 'classifier.4.bias' in weights: # triplet classification
        backbone = TripletNet(backbone=ResNet(network_cfg=resnet50_cfg, out_features=128))
        class_net = TripletClassificationNet(backbone=backbone, backbone_out_features=128, num_classes=num_classes)
        class_net.load_state_dict(weights)
    else:
        try:
            class_net =  ResNet(network_cfg=resnet50_cfg, out_features=num_classes)
            class_net.load_state_dict(weights)
        except e:
            print(e)
            class_net = torchvision.models.resnet50()
            # class_net = torchvision.models.resnet101()
            class_net.fc = torch.nn.Linear(class_net.fc.in_features, num_classes)
            class_net.load_state_dict(weights)

    # load model weights and map if was DataParallel model

    ds = DataLoader(ds, batch_size=args.batch_size)

    if torch.cuda.is_available():
        class_net = class_net.to("cuda")

    class_net = class_net.eval()
    
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

            if torch.cuda.is_available():
                input_img = input_img.to("cuda")

            if class_type == ClassType.chain_id:
                label = chain_id
            elif class_type == ClassType.hotel_id:
                label = hotel_id

            pred_label_probs = class_net(input_img).cpu()
            num_classes = pred_label_probs.shape[-1]
            pred_label = torch.argmax(pred_label_probs, dim=-1)
            gt.append(label)
            preds.append(pred_label)
            metrics = compute_classification_metrics(pred_label_probs, label.squeeze())

            del metrics['cm']
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
            # if idx == 40:
            #     break

        gt = torch.concat(gt).squeeze()
        preds = torch.concat(preds)

        metrics = {key: [metrics[key] for metrics in metricsb] for key in metricsb[0].keys()}
        metrics = {key: torch.mean(torch.vstack(metrics)).item() for key, metrics in metrics.items()}
        print(metrics)

        ax = plt.subplot()
        # plot_confusion_matrix(cm, ax)
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("-d", "--dataset-path",type=Path, default="data/dataset/hotel_test_chain.h5")
    parser.add_argument('--batch-size', type=int, default=128)
    main(parser.parse_args())