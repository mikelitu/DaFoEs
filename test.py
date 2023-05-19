import torch
from datasets.augmentations import Compose, CentreCrop, SquareResize, BrightnessContrast, Normalize, ArrayToTensor
from models.force_estimator import ForceEstimator
from path import Path
from datasets.test_dataset import TestDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from train import vit_predict_force_state, vit_predict_force_visu
import pickle
import os
from utils import dtw, frdist

parser = argparse.ArgumentParser(description="Script to test the different models for ForceEstimation variability",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--architecture", choices=['cnn', 'vit', 'fc'], default='vit', help='The chosen architecture to test')
parser.add_argument("--state", action='store_true', help='Include the state')
parser.add_argument("--train-type", type=str, default='random', help='The training type of the chosen model')
parser.add_argument("--data-dir", type=str, help="Directory of the data")
parser.add_argument("--save-dir", default='results', type=str, help='Save directory for the metrics and predictions')
parser.add_argument("--save", action='store_true', help='Save metrics and predictions for further analysis')
parser.add_argument("--recurrency", action='store_true')
parser.add_argument("--include-depth", action='store_true')
parser.add_argument("--chua", action="store_true")
parser.add_argument("--att-type", default=None, help="Additional attention values")


def load_test_experiment(architecture: str, include_depth: bool, include_state: bool = True, recurrency: bool = False,  train_mode: str = "random", att_type: str = None):
    train_modes = ["random", "color", "geometry", "structure", "stiffness", "position"]
    assert architecture.lower() in ["vit", "cnn", "fc"], "The architecture has to be either 'vit' or 'cnn', '{}' is not valid".format(architecture)
    assert train_mode in train_modes, "'{}' is not an available training mode. The available training mode are: {}".format(train_mode, train_modes)

    model = ForceEstimator(architecture,
                           recurrency=recurrency,
                           pretrained=False,
                           include_depth=include_depth,
                           att_type=att_type)

    if architecture.lower() in ["rnn", "rnn-bam"]:
        recurrency_size = 5
    else:
        recurrency_size = 1

    # Find the corresponding checkpoint
    print("LOADING EXPERIMENT [==>  ]")
    checkpoints_root = Path('/home/md21local/mreyzabal/checkpoints/img2force')
    
    if architecture.lower() == 'fc':
        checkpoints = checkpoints_root/"{}/{}".format(architecture, train_mode)
    else:
        checkpoints = checkpoints_root/"{}/{}/{}_{}".format("rgbd" if include_depth else "rgb", architecture, "visu_state" if include_state else "visu", train_mode)
    
    print('The checkpoints are loaded from: {}'.format(sorted(checkpoints.dirs())[-1]))   
    checkpoint_dir = sorted(checkpoints.dirs())[-1]/'model_best.pth.tar'
    print("LOADING EXPERIMENT [===> ]")
    print("Loading weights...")
    checkpoint = torch.load(checkpoint_dir)
    model.load_state_dict(checkpoint['state_dict'], strict=True)

    print("LOADING EXPERIMENT [====>]")
    print("Loading test dataset for corresponding model...")

    root_dir = Path('/home/md21local/visu_depth_haptic_data')

    test_dirs = {
        'random': 'DD_P_D_R_2_1',
        'color': 'EE_P_D_R_L1',
        'geometry': 'DE__S_R_2_3',
        'stiffness': 'DE_P_D_R_3_4',
        'structure': 'ED_P_D_R_C',
        'position': 'E_S_S_R_R2'
    }

    share_dir = 'E_P_S_P_1_1'

    test_data_dir = test_dirs[train_mode]
    data_dir = root_dir/test_data_dir
    print('Loading data from {}'.format(data_dir))

    test_data = load_from_folder(data_dir, limit=450, recurrency_size=recurrency_size, include_depth=include_depth)

    print('Loading data from shared dataset {}'.format(root_dir/share_dir))
    share_data = load_from_folder(root_dir/share_dir, limit=450, recurrency_size=recurrency_size, include_depth=include_depth)

    return model, test_data, share_data


def run_test_experiment(architecture: str, transforms: Compose, include_depth: bool, include_state: bool = True, train_mode: str = "random"):

    test_predictions, shared_predictions = [], []
    test_metrics, shared_metrics = [], []
    test_forces, shared_forces = [], []

    # Loading the necessary data
    model, test_data, shared_data = load_test_experiment(architecture, include_depth=include_depth, include_state=include_state, train_mode=train_mode)

    model.to(torch.device("cuda"))

    model.eval()

    for j in tqdm(range(len(test_data))):
        inp = test_data[j]
        imgs = [load_as_float(img) for img in inp['img']]
        if include_depth:
            depths = [load_as_float(d) for d in inp['depth']]
        else:
            depths = inp['depth']
        imgs, depths, state, force = transforms(imgs, depths, inp['state'], inp['force'])
        if include_depth:
            depths = [process_depth(d) for d in depths]
            imgs = [torch.cat([img, depth], axis=0) for img, depth in zip(imgs, depths)]
        
        imgs = [img.unsqueeze(0).cuda() for img in imgs]
        state = torch.from_numpy(state).float().cuda()
        force = torch.from_numpy(force).float().cuda()

        if architecture.lower() in ['vit', 'vit-base', 'vit-dist'] and include_state:
            pred_force = vit_predict_force_state(model, imgs[0], state, force)
        elif architecture.lower() in ['vit', 'vit-base', 'vit-dist'] and not include_state:
            pred_force = vit_predict_force_visu(model, imgs[0])
        elif architecture.lower() in ['cnn', 'cnn-bam'] and include_state:
            pred_force = model(imgs[0], state)
        elif architecture.lower() in ['cnn', 'cnn-bam'] and not include_state:
            pred_force = model(imgs[0])
        elif architecture.lower() in ['rnn', 'rnn-bam'] and include_state:
            pred_force = model(imgs, state.unsqueeze(0))
        elif architecture.lower() in ['rnn', 'rnn-bam'] and not include_state:
            pred_force = model(imgs)
        else:
            state = state.view(-1, 26)
            pred_force = model(state)
        
        rmse = torch.sqrt(((force - pred_force) ** 2).mean())

        test_metrics.append(rmse.item())
        test_forces.append([f for f in force.detach().cpu().numpy()])
        test_predictions.append([f for f in pred_force.detach().cpu().numpy()])

    test_metrics = np.array(test_metrics)
    test_forces = np.array(test_forces).reshape(-1, 3)
    test_predictions = np.array(test_predictions).reshape(-1, 3)


    for b in tqdm(range(len(shared_data))):
        inp = test_data[b]
        imgs = [load_as_float(img) for img in inp['img']]
        if include_depth:
            depths = [load_as_float(d) for d in inp['depth']]
        else:
            depths = inp['depth']
        imgs, depths, state, force = transforms(imgs, depths, inp['state'], inp['force'])
        if include_depth:
            depths = [process_depth(d) for d in depths]
            imgs = [torch.cat([img, depth], axis=0) for img, depth in zip(imgs, depths)]
        imgs = [img.unsqueeze(0).cuda() for img in imgs]
        state = torch.from_numpy(state).float().cuda()
        force = torch.from_numpy(force).float().cuda()

        if architecture.lower() in ['vit', 'vit-base', 'vit-dist'] and include_state:
            pred_force = vit_predict_force_state(model, img, state, force)
        elif architecture.lower() in ['vit', 'vit-base', 'vit-dist'] and not include_state:
            pred_force = vit_predict_force_visu(model, img)
        elif architecture.lower() in ['cnn', 'cnn-bam'] and include_state:
            pred_force = model(imgs[0], state)
        elif architecture.lower() in ['cnn', 'cnn-bam'] and not include_state:
            pred_force = model(imgs[0])
        elif architecture.lower() in ['rnn', 'rnn-bam'] and include_state:
            pred_force = model(imgs, state.unsqueeze(0))
        elif architecture.lower() in ['rnn', 'rnn-bam'] and not include_state:
            pred_force = model(imgs)
        else:
            state = state.view(-1, 26)
            pred_force = model(state)
        
        rmse = torch.sqrt(((force - pred_force) ** 2).mean())

        shared_metrics.append(rmse.item())
        shared_forces.append([f for f in force.detach().cpu().numpy()])
        shared_predictions.append([f for f in pred_force.detach().cpu().numpy()])
    
    shared_metrics = np.array(shared_metrics)
    shared_forces = np.array(shared_forces).reshape(-1, 3)
    shared_predictions = np.array(shared_predictions).reshape(-1, 3)
    # shared_dtw = [dtw(f, p) for f, p in zip(shared_forces, shared_predictions)]

    results = {'test_rmse': test_metrics, 'test_gt': test_forces, 'test_pred': test_predictions,
               'shared_rmse': shared_metrics, 'shared_gt': shared_forces, 'shared_pred': shared_predictions}

    return results


def save_results(args, results):
    root_dir = Path('/home/md21local/img2force')
    print("The results will be saved at: {}/{}/{}".format(root_dir, args.save_dir, "rgbd" if args.include_depth else "rgb"))
    save_dir = root_dir/args.save_dir/"{}".format("rgbd" if args.include_depth else "rgb")
    save_dir.makedirs_p()
    f = open(save_dir/'{}_{}_{}.pkl'.format(args.architecture.lower(), "state" if args.state else "visu", args.train_type), 'wb')
    pickle.dump(results, f)
    f.close()
    print("Saved the results in {}/{}_{}_{}.pkl".format(args.save_dir, args.architecture.lower(), "state" if args.state else "visu", args.train_type))

@torch.no_grad()
def main():
    args = parser.parse_args()
    normalize = Normalize(
        mean = [0.45, 0.45, 0.45],
        std = [0.225, 0.225, 0.225]
    )

    transforms = Compose([
        CentreCrop(),
        SquareResize(),
        ArrayToTensor(),
        normalize
    ])

    results = run_test_experiment(args.architecture, transforms=transforms, include_depth=args.include_depth, include_state=args.state, train_mode=args.train_type)
    if args.save:
        save_results(args, results)


if __name__ == "__main__":
    main()