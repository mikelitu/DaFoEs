import torch
from datasets.augmentations import Compose, CentreCrop, SquareResize, BrightnessContrast, Normalize, ArrayToTensor
from models.force_estimator import ForceEstimator
from path import Path
from datasets.vision_state_dataset import VisionStateDataset
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import pickle
import os


parser = argparse.ArgumentParser(description="Script to test the different models for ForceEstimation variability",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", choices=["chua", "img2force", "mixed"])
parser.add_argument("--architecture", choices=['cnn', 'vit', 'fc'], default='vit', help='The chosen architecture to test')
parser.add_argument("--type", type=str, default="vs", choices=["v", "vs"], help='Include the state')
parser.add_argument("--train-type", type=str, default='random', help='The training type of the chosen model')
parser.add_argument("--save-dir", default='results', type=str, help='Save directory for the metrics and predictions')
parser.add_argument('--occlude-param', choices=["force_sensor", "robot_p", "robot_o", "robot_v", "robot_w", "robot_q", "robot_vq", "robot_tq", "robot_qd", "robot_tqd", "None"], help="choose the parameters to occlude")
parser.add_argument("--save", action='store_true', help='Save metrics and predictions for further analysis')
parser.add_argument("--recurrency", action='store_true')
parser.add_argument("--include-depth", action='store_true')
parser.add_argument("--att-type", default=None, help="Additional attention values")


def load_test_experiment(architecture: str, include_depth: bool, data: str, include_state: bool = True, recurrency: bool = False,  train_mode: str = "random", att_type: str = None):
    train_modes = ["random", "color", "geometry", "structure", "stiffness", "position"]
    assert architecture.lower() in ["vit", "cnn", "fc"], "The architecture has to be either 'vit' or 'cnn', '{}' is not valid".format(architecture)
    assert train_mode in train_modes, "'{}' is not an available training mode. The available training mode are: {}".format(train_mode, train_modes)
    assert data in ["img2force", "chua", "mixed"], "The available datasets for this case are: 'img2force' or 'chua'."
    
    if include_state:
        state_size = 54 
    else:
        state_size = 0

    model = ForceEstimator(architecture,
                           state_size=state_size,
                           recurrency=recurrency,
                           pretrained=False,
                           include_depth=include_depth,
                           att_type=att_type)

    if recurrency:
        recurrency_size = 5
    else:
        recurrency_size = 1

    # Find the corresponding checkpoint
    print("LOADING EXPERIMENT [==>  ]")
    checkpoints_root = Path('/nfs/home/mreyzabal/checkpoints/{}'.format(data))
    
    if architecture.lower() == 'fc':
        checkpoints = checkpoints_root/"{}/{}".format(architecture, "visu_state_"+train_mode)
    else:
        checkpoints = checkpoints_root/"{}/{}/{}_{}".format("rgbd" if include_depth else "rgb", "r"+architecture if recurrency else architecture, "visu_state" if include_state else "visu", train_mode)
    
    print('The checkpoints are loaded from: {}'.format(sorted(checkpoints.dirs())[-1]))   
    checkpoint_dir = sorted(checkpoints.dirs())[-1]/'checkpoint.pth.tar'
    print("LOADING EXPERIMENT [===> ]")
    print("Loading weights...")
    checkpoint = torch.load(checkpoint_dir)
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    print("LOADING EXPERIMENT [====>]")
    print("Loading test dataset for corresponding model...")

    normalize = Normalize(
        mean = [0.45, 0.45, 0.45],
        std = [0.225, 0.225, 0.225]
    )
    bright = BrightnessContrast(
        contrast=2.,
        brightness=12.
    )

    transforms = Compose([
        CentreCrop(),
        SquareResize(),
        ArrayToTensor(),
        normalize
    ]) if data=="chua" else Compose([
        CentreCrop(),
        SquareResize(),
        bright,
        ArrayToTensor(),
        normalize
    ])

    dataset = VisionStateDataset(transform=transforms, mode="test", recurrency_size=recurrency_size,
                          dataset=data, load_depths=include_depth)

    print("The length of the testing dataset is: ", len(dataset))

    dataloader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=0)

    return model, dataloader


def run_test_experiment(architecture: str, include_depth: bool, data: str, recurrency: bool = False, include_state: bool = True, train_mode: str = "random"):

    test_predictions, shared_predictions = [], []
    test_metrics, shared_metrics = [], []
    test_forces, shared_forces = [], []

    # Loading the necessary data
    model, dataloader = load_test_experiment(architecture, include_depth=include_depth, include_state=include_state, train_mode=train_mode, data=data, recurrency=recurrency)
    
    device = torch.device("cuda")
    
    model.to(device)

    model.eval()

    for i, data in enumerate(tqdm(dataloader)):
        img = [im.to(device) for im in data['img']] if recurrency else data['img'].to(device)
        
        if include_state:
            state = data['robot_state'].squeeze(1).to(device).float()
        else:
            state = None
        
        forces = data['forces'].to(device).float()

        pred_forces = model(state) if architecture=="fc" else model(img, state)
        
        rmse = torch.sqrt(((forces - pred_forces) ** 2).mean(dim=1))

        for i in range(rmse.shape[0]):
            test_metrics.append(rmse[i].item())
            test_forces.append(forces[i].detach().cpu().numpy())
            test_predictions.append(pred_forces[i].detach().cpu().numpy())

    test_metrics = np.array(test_metrics)
    test_forces = np.array(test_forces).reshape(-1, 3)
    test_predictions = np.array(test_predictions).reshape(-1, 3)

    results = {'test_rmse': test_metrics, 'test_gt': test_forces, 'test_pred': test_predictions,
               'shared_rmse': shared_metrics, 'shared_gt': shared_forces, 'shared_pred': shared_predictions}

    return results


def save_results(args, results, include_state: bool):
    root_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    print("The results will be saved at: {}/{}/{}/{}".format(root_dir, args.save_dir, args.dataset, "rgbd" if args.include_depth else "rgb"))
    save_dir = root_dir/args.save_dir/"{}/{}".format(args.dataset, "rgbd" if args.include_depth else "rgb")
    save_dir.makedirs_p()
    f = open(save_dir/'{}_{}_{}.pkl'.format("r"+args.architecture.lower() if args.recurrency else args.architecture.lower(), "state" if include_state else "visu", args.train_type), 'wb')
    pickle.dump(results, f)
    f.close()
    print("Saved the results in {}/{}_{}_{}.pkl".format(save_dir, args.architecture.lower(), "state" if include_state else "visu", args.train_type))

@torch.no_grad()
def main():
    args = parser.parse_args()

    if args.type == "vs":
        include_state = True
    else:
        include_state = False

    results = run_test_experiment(args.architecture, include_depth=args.include_depth, include_state=include_state, train_mode=args.train_type, recurrency=args.recurrency, data=args.dataset)
    if args.save:
        save_results(args, results, include_state)


if __name__ == "__main__":
    main()