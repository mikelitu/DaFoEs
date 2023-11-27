import torch
from datasets.augmentations import Compose, CentreCrop, SquareResize, BrightnessContrast, Normalize, ArrayToTensor
from models.force_estimator import ForceEstimator
from path import Path
from datasets.surgical_dataset import SurgicalDataset
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import pickle
import os
from utils import none_or_str


parser = argparse.ArgumentParser(description="Script to test the different models for ForceEstimation variability",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("data_root", type=str, help="The root directory of the data")
parser.add_argument("--dataset", choices=["dvrk", "dafoes", "mixed"])
parser.add_argument("--architecture", choices=['cnn', 'vit', 'fc'], default='vit', help='The chosen architecture to test')
parser.add_argument("--type", type=str, default="vs", choices=["v", "vs"], help='Include the state')
parser.add_argument("--train-type", type=str, default='random', help='The training type of the chosen model')
parser.add_argument("--save-dir", default='results', type=str, help='Save directory for the metrics and predictions')
parser.add_argument('--occlude-param', choices=["force_sensor", "robot_p", "robot_o", "robot_v", "robot_w", "robot_q", "robot_vq", "robot_tq", "robot_qd", "robot_tqd", "None"], help="choose the parameters to occlude")
parser.add_argument("--save", action='store_true', help='Save metrics and predictions for further analysis')
parser.add_argument("--recurrency", action='store_true')


def load_test_experiment(architecture: str, data: str, data_root: Path, include_state: bool = True, recurrency: bool = False,  train_mode: str = "random", occ_param: str = None):
    train_modes = ["random", "color", "geometry", "structure", "stiffness", "position"]
    assert architecture.lower() in ["vit", "cnn", "fc"], "The architecture has to be either 'vit' or 'cnn', '{}' is not valid".format(architecture)
    assert train_mode in train_modes, "'{}' is not an available training mode. The available training mode are: {}".format(train_mode, train_modes)
    assert data in ["dafoes", "dvrk", "mixed"], "The available datasets for this case are: 'dafoes', 'dvrk' or 'mixed'."
    
    if include_state:
        state_size = 54 
    else:
        state_size = 0

    model = ForceEstimator(architecture,
                           state_size=state_size,
                           recurrency=recurrency,
                           pretrained=False)

    if recurrency:
        recurrency_size = 5
    else:
        recurrency_size = 1

    # Find the corresponding checkpoint
    print("LOADING EXPERIMENT [==>  ]")
    root = os.path.dirname(os.path.abspath(__file__))
    checkpoints_root = Path("{}/checkpoints/{}".format(root, data))
    
    if architecture.lower() == 'fc':
        if occ_param is None:
            checkpoints = checkpoints_root/"{}/{}".format(architecture, "visu_state_"+train_mode)
        else:
            checkpoints = checkpoints_root/"{}/{}/{}".format(architecture, occ_param, "visu_state_"+train_mode)
    else:
        if occ_param is None:
            checkpoints = checkpoints_root/"rgb/{}/{}_{}".format("r"+architecture if recurrency else architecture, "visu_state" if include_state else "visu", train_mode)
        else:
            checkpoints = checkpoints_root/"rgb/{}/{}/{}_{}".format("r"+architecture if recurrency else architecture, occ_param, "visu_state" if include_state else "visu", train_mode)
    
    # print('The checkpoints are loaded from: {}'.format(sorted(checkpoints.dirs())[-1]))   
    # checkpoint_dir = sorted(checkpoints.dirs())[-1]/'checkpoint.pth.tar'
    print("LOADING EXPERIMENT [===> ]")
    print("Loading weights...")
    # checkpoint = torch.load(checkpoint_dir)
    # model.load_state_dict(checkpoint['state_dict'], strict=False)

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
    ]) if data=="dvrk" else Compose([
        CentreCrop(),
        SquareResize(),
        bright,
        ArrayToTensor(),
        normalize
    ])

    dataset = SurgicalDataset(root=data_root, transform=transforms, recurrency_size=recurrency_size)

    print("The length of the testing dataset is: ", len(dataset))

    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

    return model, dataloader


def run_test_experiment(architecture: str, data: str, data_root: Path, recurrency: bool = False, include_state: bool = True, train_mode: str = "random", occ_param: str = None):

    predictions = []

    # Loading the necessary data
    model, dataloader = load_test_experiment(architecture, data=data, data_root=data_root, include_state=include_state, train_mode=train_mode, recurrency=recurrency, occ_param=occ_param)
    
    device = torch.device("cuda")
    
    model.to(device)

    model.eval()

    for i, data in enumerate(tqdm(dataloader)):
        img = [im.to(device) for im in data['img']] if recurrency else data['img'].to(device)
        
        if include_state:
            state = data['robot_state'].squeeze(1).to(device).float()
        else:
            state = None

        pred_force = model(state) if architecture=="fc" else model(img, state)
        
        for i in range(pred_force.shape[0]):
            predictions.append(pred_force[i].detach().cpu().numpy())

    predictions = np.array(predictions).reshape(-1, 3)

    results = {'pred': predictions}

    return results


def save_results(args, results, include_state: bool, occ_param: str = None):
    root_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    if occ_param is None:
        print("The results will be saved at: {}/{}/{}/real_surgery".format(root_dir, args.save_dir, args.dataset))
        save_dir = root_dir/args.save_dir/"{}/real_surgery".format(args.dataset)
        save_dir.makedirs_p()
        f = open(save_dir/'{}_{}_{}.pkl'.format("r"+args.architecture.lower() if args.recurrency else args.architecture.lower(), "state" if include_state else "visu", args.train_type), 'wb')
    else:
        print("The results will be saved at: {}/{}/{}/real_surgery/{}".format(root_dir, args.save_dir, args.dataset, occ_param))
        save_dir = root_dir/args.save_dir/"{}/real_surgery/{}".format(args.dataset, occ_param)
        save_dir.makedirs_p()
        f = open(save_dir/'{}_{}_{}.pkl'.format("r"+args.architecture.lower() if args.recurrency else args.architecture.lower(), "state" if include_state else "visu", args.train_type), 'wb')

    pickle.dump(results, f)
    f.close()
    print("Saved the results in {}/{}_{}_{}.pkl".format(save_dir, args.architecture.lower(), "state" if include_state else "visu", args.train_type))

def get_metrics(list_of_results: list[dict]):
    # List of resuls contains a list of dicttionaries with keys: "dafoes_rmse", "dafoes_gt", "dafoes_pred", "dvrk_rmse", "dvrk_gt", "dvrk_pred"
    tmp_results = {}
    keys = list_of_results[0].keys()
    for r in list_of_results:
        for key in keys:
            if key in tmp_results:
                tmp_results[key].append(r[key])
            else:
                tmp_results[key] = [r[key]]
    
    results = {}
    for key in keys:
        result = tmp_results[key]
        mean_r = np.mean(result, axis=0)
        std_r = np.std(result, axis=0)
        results[f"{key}_mean"] = mean_r
        results[f"{key}_std"] = std_r
    
    return results

@torch.no_grad()
def main():
    args = parser.parse_args()

    if args.type == "vs":
        include_state = True
    else:
        include_state = False

    occ_param = none_or_str(args.occlude_param)
    num_experiments = 1

    print("Loading data from {}".format(args.data_root))
    list_of_results = [run_test_experiment(args.architecture, data=args.dataset, data_root=args.data_root, include_state=include_state, train_mode=args.train_type, recurrency=args.recurrency, occ_param=occ_param) for _ in range(num_experiments)]
    
    results = get_metrics(list_of_results)

    print(results)
    
    if args.save:
        save_results(args, results, include_state, occ_param=occ_param)


if __name__ == "__main__":
    main()