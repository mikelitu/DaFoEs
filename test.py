import torch
from datasets.augmentations import Compose, CentreCrop, SquareResize, Normalize, ArrayToTensor
from models.force_estimator_2d import ForceEstimatorV, ForceEstimatorVS
from models.force_estimator_transformers import ViT
from models.force_estimator_transformers_base import BaseViT
from path import Path
from datasets.vision_state_dataset import normalize_labels, load_as_float
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from train import vit_predict_force_state, vit_predict_force_visu
from train_cnn import cnn_predict_force_state, cnn_predict_force_visu
import pickle
import os
from utils import dtw, frdist

parser = argparse.ArgumentParser(description="Script to test the different models for ForceEstimation variability",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--architecture", choices=['cnn', 'vit', 'cnn-bam', 'vit-base'], default='vit', help='The chosen architecture to test')
parser.add_argument("--state", action='store_true', help='Include the state')
parser.add_argument("--train-type", type=str, default='random', help='The training type of the chosen model')
parser.add_argument("--data-dir", type=str, help="Directory of the data")
parser.add_argument("--save-dir", default='results', type=str, help='Save directory for the metrics and predictions')
parser.add_argument("--save", action='store_true', help='Save metrics and predictions for further analysis')


def load_from_folder(folder: Path, limit: int = 1000):
    folder = Path(folder)
    labels = np.array(pd.read_csv(folder/'labels.csv'))
    norm_labels = normalize_labels(labels)

    inputs = []
    nlabels = labels.shape[0] // len(folder.files('*.png'))
    step = 7

    breaking_limit = min(len(folder.files('*.png')) - 40, limit)

    print('Loading a total of {} images'.format(breaking_limit - 20))

    for i, file in enumerate(sorted(folder.files('*.png'))):
        if i == breaking_limit:
            break
        if i < 20:
            continue
            
        data = {}
        img = load_as_float(file)
        state = norm_labels[i*nlabels:(i*nlabels) + step, :-6]
        force = labels[i*nlabels:(i*nlabels) + step, -6:]
        data['img'] = img
        data['state'] = state
        data['force'] = 0.25 * force
        inputs.append(data)
    
    return inputs


def load_test_experiment(architecture: str, include_state: bool = True,  train_mode: str = "random"):
    train_modes = ["random", "color", "geometry", "structure", "stiffness", "position"]
    assert architecture.lower() in ["vit", "cnn", "vit-base", "cnn-bam"], "The architecture has to be either 'vit' or 'cnn', '{}' is not valid".format(architecture)
    assert train_mode in train_modes, "'{}' is not an available training mode. The available training mode are: {}".format(train_mode, train_modes)
    
    
    
    print("Experiment variables: architecture -> {}, include_state -> {} & train_mode -> {}".format(architecture, include_state, train_mode))

    if architecture.lower() == "vit":
        print("LOADING EXPERIMENT [=>   ]")
        print("Chosen model is Vision Transformer (ViT) with token sampling {}".format("vision+state" if include_state else "vision only"))
        # Choosing the model
        model = ViT(
            image_size = 256,
            patch_size = 16,
            num_classes = 6,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1,
            max_tokens_per_depth = (256, 128, 64, 32, 16, 8),
            state_include = include_state
        )
    
    elif architecture.lower() == 'vit-base':
        print("LOADING EXPERIMENT [=>   ]")
        print("Chosen model is Vision Transformer (ViT) {}".format("vision+state" if include_state else "vision only"))
        model = BaseViT(
            image_size = 256,
            patch_size = 16,
            num_classes = 6,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1,
            state_include = include_state
        )

    elif architecture.lower() == 'cnn-bam':
        print("LOADING EXPERIMENT [=>   ]")
        print("Chosen model is ResNet50 (CNN) {}".format("vision+state" if include_state else "vision only"))
        # Choosing the model
        model = ForceEstimatorVS(rs_size=25, num_layers=50, pretrained=False, att_type='BAM') if include_state else ForceEstimatorV(num_layers=50, pretrained=False, att_type='BAM')

    else:
        print("LOADING EXPERIMENT [=>   ]")
        print("Chosen model is ResNet50 (CNN) {}".format("vision+state" if include_state else "vision only"))
        # Choosing the model
        model = ForceEstimatorVS(rs_size=25, num_layers=50, pretrained=False) if include_state else ForceEstimatorV(num_layers=50, pretrained=False)


    # Find the corresponding checkpoint
    print("LOADING EXPERIMENT [==>  ]")
    checkpoints_root = Path('/nfs/home/mreyzabal/checkpoints/img2force')
    checkpoints = checkpoints_root/"{}/{}_{}".format(architecture, "visu_state" if include_state else "visu", train_mode)
    print('The checkpoints are loaded from: {}'.format(sorted(checkpoints.dirs())[-1]))   
    checkpoint_dir = sorted(checkpoints.dirs())[-1]/'checkpoint.pth.tar'
    print("LOADING EXPERIMENT [===> ]")
    print("Loading weights...")
    checkpoint = torch.load(checkpoint_dir)
    model.load_state_dict(checkpoint['state_dict'], strict=True)

    print("LOADING EXPERIMENT [====>]")
    print("Loading test dataset for corresponding model...")

    root_dir = Path('/nfs/home/mreyzabal/visu_haptic_data')

    test_dirs = {
        'random': 'DE_P_D_P_R2',
        'color': 'EE_P_D_R_L1',
        'geometry': 'D_S_S_P_L2',
        'stiffness': 'D_P_S_R_R1',
        'structure': 'ED_P_D_R_C',
        'position': 'E_S_S_R_R2'
    }

    share_dir = 'E_P_S_P_C'

    test_data_dir = test_dirs[train_mode]
    data_dir = root_dir/test_data_dir
    print('Loading data from {}'.format(data_dir))

    test_data = load_from_folder(data_dir, 620)

    print('Loading data from shared dataset {}'.format(root_dir/share_dir))
    share_data = load_from_folder(root_dir/share_dir, 620)

    return model, test_data, share_data


def run_test_experiment(architecture: str, transforms, include_state: bool = True, train_mode: str = "random"):

    test_predictions, shared_predictions = [], []
    test_metrics, shared_metrics = [], []
    test_forces, shared_forces = [], []

    # Loading the necessary data
    model, test_data, shared_data = load_test_experiment(architecture, include_state, train_mode)

    model.to(torch.device("cuda"))

    model.eval()

    for j in tqdm(range(len(test_data))):
        d = test_data[j]
        img = transforms([d['img']])[0].unsqueeze(0).cuda()
        state = torch.from_numpy(d['state']).unsqueeze(0).float().cuda()
        force = torch.from_numpy(d['force']).unsqueeze(0).float().cuda()

        if architecture.lower() in ['vit', 'vit-base'] and include_state:
            pred_force = vit_predict_force_state(model, img, state, force)
        elif architecture.lower() in ['vit', 'vit-base'] and not include_state:
            pred_force = vit_predict_force_visu(model, img)
        elif architecture.lower() in ['cnn', 'cnn-bam'] and include_state:
            pred_force = cnn_predict_force_state(model, img, state, force)
        elif architecture.lower() in ['cnn', 'cnn-bam'] and not include_state:
            pred_force = cnn_predict_force_visu(model, img)
        
        rmse = torch.sqrt(((force - pred_force) ** 2).mean()) if include_state else torch.sqrt(((force.mean(axis=1) - pred_force) ** 2).mean())

        test_metrics.append(rmse.item())
        test_forces.append([f for f in force.squeeze(0).detach().cpu().numpy()])
        test_predictions.append([f for f in pred_force.squeeze(0).detach().cpu().numpy()] if include_state else pred_force.squeeze(0).detach().cpu().numpy())
    
    test_metrics = np.array(test_metrics)
    test_forces = np.array(test_forces).reshape(-1, 6)[:, :3] if include_state else np.array(test_forces).mean(axis=0)[:, :3]
    test_predictions = np.array(test_predictions).reshape(-1, 6)[:, :3] if include_state else np.array(test_predictions)[:, :3]
    test_dtw = [dtw(f, p) for f, p in zip(test_forces, test_predictions)]


    for j in tqdm(range(len(shared_data))):
        d = shared_data[j]
        img = transforms([d['img']])[0].unsqueeze(0).cuda()
        state = torch.from_numpy(d['state']).unsqueeze(0).float().cuda()
        force = torch.from_numpy(d['force']).unsqueeze(0).float().cuda()

        if architecture.lower() in ['vit', 'vit-base'] and include_state:
            pred_force = vit_predict_force_state(model, img, state, force)
        elif architecture.lower() in ['vit', 'vit-base'] and not include_state:
            pred_force = vit_predict_force_visu(model, img)
        elif architecture.lower() in ['cnn', 'cnn-bam'] and include_state:
            pred_force = cnn_predict_force_state(model, img, state, force)
        elif architecture.lower() in ['cnn', 'cnn-bam'] and not include_state:
            pred_force = cnn_predict_force_visu(model, img)
        
        rmse = torch.sqrt(((force - pred_force) ** 2).mean()) if include_state else torch.sqrt(((force.mean(axis=1) - pred_force) ** 2).mean())

        shared_metrics.append(rmse.item())
        shared_forces.append([f for f in force.squeeze(0).detach().cpu().numpy()])
        shared_predictions.append([f for f in pred_force.squeeze(0).detach().cpu().numpy()] if include_state else pred_force.squeeze(0).detach().cpu().numpy())
    
    shared_metrics = np.array(shared_metrics)
    shared_forces = np.array(shared_forces).reshape(-1, 6)[:, :3] if include_state else np.array(shared_forces).mean(axis=0)[:, :3]
    shared_predictions = np.array(shared_predictions).reshape(-1, 6)[:, :3] if include_state else np.array(shared_predictions)[:, :3]
    shared_dtw = [dtw(f, p) for f, p in zip(shared_forces, shared_predictions)]

    results = {'test_rmse': test_metrics, 'test_dtw': test_dtw, 'test_gt': test_forces, 'test_pred': test_predictions,
               'shared_rmse': shared_metrics, 'shared_dtw': shared_dtw, 'shared_gt': shared_forces, 'shared_pred': shared_predictions}

    return results


def save_results(args, results):
    root_dir = Path('/nfs/home/mreyzabal/img2force')
    print("The results will be saved at: {}/{}".format(root_dir, args.save_dir))
    save_dir = root_dir/args.save_dir
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

    results = run_test_experiment(args.architecture, transforms, args.state, args.train_type)
    if args.save:
        save_results(args, results)


if __name__ == "__main__":
    main()