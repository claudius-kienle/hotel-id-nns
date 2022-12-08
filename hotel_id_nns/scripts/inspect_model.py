import argparse
from pathlib import Path
from matplotlib import pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
from hotel_id_nns.nn.datasets.chain_dataset_h5 import H5ChainDataset
from hotel_id_nns.utils.plotting import plot_confusion_matrix
from hotel_id_nns.utils.pytorch import load_model_weights

root_dir = Path(__file__).parent.parent.parent

def main(args):
    config = {
        "num_chain_id_classes": 88
    }
    ds = H5ChainDataset(annotations_file_path=root_dir / args.dataset_path, config=config)

    class_net =  torchvision.models.resnet50() # ,num_classes=
    class_net.fc = torch.nn.Linear(512 * torchvision.models.resnet.Bottleneck.expansion, config['num_chain_id_classes'])

    # load model weights and map if was DataParallel model
    class_net.load_state_dict(load_model_weights(root_dir / args.model_path))

    ds = DataLoader(ds, batch_size=args.batch_size)
    
    # generate predictions on dataset
    gt = []
    preds = []
    for sample in tqdm(ds):
        input_img, chain_id = sample
        pred_chain_id_probs = class_net(input_img)
        num_classes = pred_chain_id_probs.shape[-1]
        pred_chain_id = torch.argmax(pred_chain_id_probs, dim=-1)
        gt.append(chain_id)
        preds.append(pred_chain_id)

    gt = torch.concat(gt)
    preds = torch.concat(preds)

    indices = num_classes * gt + preds
    cm = torch.bincount(indices, minlength=num_classes ** 2).reshape((num_classes, num_classes))
    plot_confusion_matrix(cm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("-d", "--dataset-path",type=Path, default="data/dataset/hotel_test_chain.h5")
    parser.add_argument('--batch-size', type=int, default=32)
    main(parser.parse_args())