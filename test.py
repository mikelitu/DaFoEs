import torch
from datasets.augmentations import Compose, CentreCrop, SquareResize, Normalize, ArrayToTensor
from models.force_estimator_2d import ForceEstimatorV, ForceEstimatorVS
from models.force_estimator_transformers import ViT
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

parser = argparse.ArgumentParser(description="Script to test the different models for ForceEstimation variability",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--architecture", choices=['cnn', 'vit'], default='vit', help='The chosen architecture to test')
parser.add_argument("--state", action='store_true', help='Include the state')
parser.add_argument("--train-type", type=str, default='random', help='The training type of the chosen model')
parser.add_argument("--save-dir", default='results', type=str, help='Save directory for the metrics and predictions')
parser.add_argument("--save", action='store_true', help='Save metrics and predictions for further analysis')


def load_from_folder(folder: Path, limit: int = 1000):
    folder = Path(folder)
    labels = np.array(pd.read_csv(folder/'labels.csv'))
    norm_labels = normalize_labels(labels)
    forces = labels[:, -6:]

    inputs = []
    nlabels = labels.shape[0] // len(folder.files('*.png'))
    step = 7

    breaking_limit = min(len(folder.files('*.png')) - 40, limit)

    print('Loading a total of {} images'.format(breaking_limit))

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
        data['force'] = 0.1 * force
        inputs.append(data)
    
    return inputs


def load_test_experiment(architecture: str, include_state: bool = True,  train_mode: str = "random"):
    train_modes = ["random", "color", "geometry", "structure", "stiffness"]
    assert architecture.lower() in ["vit", "cnn"], "The architecture has to be either 'vit' or 'cnn', '{}' is not valid".format(architecture)
    assert train_mode in train_modes, "'{}' is not an available training mode. The available training mode are: {}".format(train_mode, train_modes)
    
    checkpoints_root = Path('/home/md21local/mreyzabal/checkpoints/img2force')
    checkpoints_dir = {
        'random': 'visu_state',
        'color': 'visu_state_color',
        'geometry': 'visu_state_geometry',
        'structure': 'visu_state_structure',
        'stiffness': 'visu_state_stiffness'
    }
    
    print("Experiment variables: architecture -> {}, include_state -> {} & train_mode -> {}".format(architecture, include_state, train_mode))

    if architecture.lower() == "vit":
        print("LOADING EXPERIMENT [=>   ]")
        print("Chosen model is Vision Transformer (ViT) {}".format("vision+state" if include_state else "vision only"))
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
        
        # Find the corresponding checkpoint
        print("LOADING EXPERIMENT [==>  ]")
        vit_checkpoints = checkpoints_root/'vit/{}'.format(checkpoints_dir[train_mode] if include_state else 'visu')
        print('The checkpoints are loaded from: {}'.format(sorted(vit_checkpoints.dirs())[-1]))   
        checkpoint_dir = sorted(vit_checkpoints.dirs())[-1]/'checkpoint.pth.tar'
    
    else:
        print("LOADING EXPERIMENT [=>   ]")
        print("Chosen model is ResNet18 (CNN) {}".format("vision+state" if include_state else "vision only"))
        # Choosing the model
        model = ForceEstimatorVS(rs_size=25, final_layer=30) if include_state else ForceEstimatorV(final_layer=6)

        # Find the corresponding checkpoint
        print("LOADING EXPERIMENT [==>  ]")
        cnn_checkpoints = checkpoints_root/'vit/{}'.format(checkpoints_dir[train_mode] if include_state else 'visu')
        print('The checkpoints are loaded from: {}'.format(sorted(cnn_checkpoints.dirs())[-1]))   
        checkpoint_dir = sorted(cnn_checkpoints.dirs())[-1]/'checkpoint.pth.tar'

    print("LOADING EXPERIMENT [===> ]")
    print("Loading weights...")
    checkpoint = torch.load(checkpoint_dir)
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    print("LOADING EXPERIMENT [====>]")
    print("Loading test dataset for corresponding model...")

    data_dir = Path('/home/md21local/test_force_visu_data')

    test_dirs = {
        'random': 'dragon_20_pink_single-layer_push',
        'color': 'eco_30_red_single-layer_push',
        'geometry': 'eco_30_pink_sphere_push',
        'stiffness': 'dragon_20_pink_single-layer_push',
        'structure': 'eco_30_red_dragon_20_pink_double-sphere_push'
    }
    test_data_dir = test_dirs[train_mode] if include_state else test_dirs['random']
    data_dir = data_dir/test_data_dir
    print('Loading data from {}'.format(data_dir))

    data = load_from_folder(data_dir, 600)

    return model, data


def run_test_experiment(architecture: str, transforms, include_state: bool = True, train_mode: str = "random"):

    predictions = []
    metrics = []
    forces = []

    # Loading the necessary data
    model, data = load_test_experiment(architecture, include_state, train_mode)

    model.to(torch.device("cuda"))

    model.eval()

    for j in tqdm(range(len(data))):
        d = data[j]
        img = transforms([d['img']])[0].unsqueeze(0).cuda()
        state = torch.from_numpy(d['state']).unsqueeze(0).float().cuda()
        force = torch.from_numpy(d['force']).unsqueeze(0).float().cuda()

        if architecture.lower() == "vit" and include_state:
            pred_force = vit_predict_force_state(model, img, state, force)
        elif architecture.lower() == 'vit' and not include_state:
            pred_force = vit_predict_force_visu(model, img)
        elif architecture.lower() == 'cnn' and include_state:
            pred_force = cnn_predict_force_state(model, img, state, force)
        elif architecture.lower() == 'cnn' and not include_state:
            pred_force = cnn_predict_force_visu(model, img)
        
        rmse = torch.sqrt(((force - pred_force) ** 2).mean()) if include_state else torch.sqrt(((force.mean(axis=1) - pred_force) ** 2).mean())

        metrics.append(rmse.item())
        forces.append([f for f in force.squeeze(0).detach().cpu().numpy()])
        predictions.append([f for f in pred_force.squeeze(0).detach().cpu().numpy()] if include_state else pred_force.squeeze(0).detach().cpu().numpy())
    
    metrics = np.array(metrics)
    forces = np.array(forces).reshape(-1, 6) if include_state else np.array(forces).mean(axis=0)
    predictions = np.array(predictions).reshape(-1, 6) if include_state else np.array(predictions)

    results = {'rmse': metrics, 'gt': forces, 'pred': predictions}

    return results


def save_results(args, results):
    print("The results will be saved at: {}".format(args.save_dir))
    save_dir = Path(args.save_dir)
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