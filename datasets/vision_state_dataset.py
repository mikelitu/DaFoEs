import torch
import os
from typing import Dict, List

from torch.utils.data import Dataset
import imageio
import numpy as np
import random
from path import Path
import pandas as pd
from PIL import ImageFile, Image
from datasets.utils import save_metric, load_metrics, RGBtoD

ImageFile.LOAD_TRUNCATED_IMAGES = True

def read_labels(label_file):
    """
    Read the txt file containing the robot state and reshape it to a meaningful vector to pair with the
    video frames.
    Parameters
    ----------
    root : str
        The root directory where the label files are
    name: str
        The name of the label file with the .txt extension
    Returns
    -------
    robot_state : ndarray
        A matrix with the 54 dimensional robot state vector for every frame in the video. The information is as follows:
        | 0 -> Time (t)
        | 1 to 6 -> Force sensor reading (fx, fy, fz, tx, ty, tz)
        | 7 to 19 -> Dvrk estimation task position, orientation, linear and angular velocities (px, py, pz, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz)
        | 20 to 26 -> Joint angles (q1, q2, q3, q4, q5, q6, q7)
        | 27 to 33 -> Joint velocities (vq1, vq2, vq3, vq4, vq5, vq6, vq7)
        | 44 to 40 -> Joint torque (tq1, tq2, tq3, tq4, tq5, tq6, tq7)
        | 41 to 47 -> Desired joint angle (q1d, q2d, q3d, q4d, q5d, q6d, q7d)
        | 48 to 54 -> Desired joint torque (tq1d, tq2d, tq3d, tq4d, tq5d, tq6d, tq7d)
        | 55 to 57 -> Estimated end effector force (psm_fx, psm_fy, psm_fz)
    """

    with open(label_file, "r") as file_object:
        lines = file_object.read().splitlines()
        robot_state = []
        for line in lines:
            row = []
            splitted = line.split(",")
            _ = [row.append(float(f)) for f in splitted]
            robot_state.append(row)

    robot_state = np.array(robot_state).astype(np.float32)
    state = robot_state[:, 1:55]
    force = robot_state[:, 55:58]

    return state, force

def dafoes_2_dvrk(labels):
    new_labels = np.zeros((labels.shape[0], 54))
    # Robot position and orientation
    new_labels[:, 6:13] = labels[:, :7]
    new_labels[:, 19:25] = labels[:, 7:13]
    new_labels[:, 40:46] = labels[:, 20:26]
    new_labels[:, 47:54] = labels[:, 13:20]

    return new_labels


def load_as_float(path: Path) -> np.ndarray:
    return  imageio.imread(path)[:,:, :3].astype(np.float32)

def load_depth(path: Path) -> np.ndarray:
    return np.array(Image.open(path)).astype(np.uint16).astype(np.float32)

def process_depth(rgb_depth: torch.Tensor) -> torch.Tensor:
    depth = torch.zeros((1, rgb_depth.shape[1], rgb_depth.shape[2]))
    for i in range(rgb_depth.shape[1]):
        for j in range(rgb_depth.shape[2]):
            pixel = RGBtoD(rgb_depth[0, i, j].item(), rgb_depth[1, i, j].item(), rgb_depth[2, i, j].item())
            depth[:, i, j].item = pixel
    
    return (depth.float() - depth.mean()) / depth.std()

class VisionStateDataset(Dataset):
    """A dataset to load data from dfferent folders that are arranged this way:
        root/scene_1/000.png
        root/scene_1/001.png
        ...
        root/scene_1/labels.csv
        root/scene_2/000.png
        .
        transform functions takes in a list images and a numpy array representing the intrinsics of the camera and the robot state
    """

    def __init__(self, recurrency_size=5, load_depths=True, max_depth=25., mode="train", transform=None, seed=0, train_type="random",
                 occlude_param=None, dataset="dafoes"):
        
        assert dataset in ["dafoes", "dvrk", "mixed"], "The only available datasets are dafoes, dvrk or mixed"
        assert mode in ["train", "val", "test"], "There is only 3 modes for the dataset: train, validation or test"

        np.random.seed(seed)
        random.seed(seed)

        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        root = Path(root)
        self.dataset = dataset

        if dataset == "dafoes":
            data_root_dafoes = root/"visu_depth_haptic_data"
            self.data_root_dafoes = data_root_dafoes
        elif dataset == "dvrk":
            data_root_dvrk = root/"experiment_data"
            self.data_root_dvrk = data_root_dvrk
        else:
            data_root_dafoes = root/"visu_depth_haptic_data"
            data_root_dvrk = root/"experiment_data"
            self.data_root_dafoes = data_root_dafoes
            self.data_root_dvrk = data_root_dvrk

        self.occlusion = {"force_sensor": [0, 6],
                          "robot_p": [6, 9],
                          "robot_o": [9, 13],
                          "robot_v": [13, 16],
                          "robot_w": [16, 19],
                          "robot_q": [19, 26],
                          "robot_vq": [26, 33],
                          "robot_tq": [33, 40],
                          "robot_qd": [40, 47],
                          "robot_tqd": [47, 54]
                        }
        
        if dataset == "dafoes":
            scene_list_path = self.data_root_dafoes/"{}.txt".format(mode) if train_type=="random" else self.data_root_dafoes/"{}_{}.txt".format(mode, train_type)
            self.scenes = [self.data_root_dafoes/folder[:-1] for folder in open(scene_list_path)][:-1]
        elif dataset == "dvrk":
            scene_list_path = self.data_root_dvrk/"{}.txt".format(mode)
            self.folder_index = [folder.split('_')[-1].rstrip('\n') for folder in open(scene_list_path)]
        else:
            scene_list_path = self.data_root_dafoes/"{}.txt".format(mode) if train_type=="random" else self.data_root_dafoes/"{}_{}.txt".format(mode, train_type)
            self.scenes = [self.data_root_dafoes/folder[:-1] for folder in open(scene_list_path)][:-1]
            scene_list_path = self.data_root_dvrk/"{}.txt".format(mode)
            self.folder_index = [folder.split('_')[-1].rstrip('\n') for folder in open(scene_list_path)]

        self.transform = transform
        self.mode = mode
        self.load_depths = load_depths
        self.max_depth = max_depth
        self.recurrency_size = recurrency_size
        self.occlude_param = occlude_param
        self.crawl_folders()
        
        
    def crawl_folders(self):
        
        samples = []
        
        if self.dataset == "dafoes":
            samples = self.load_dafoes(samples)
        
        elif self.dataset == "dvrk":
            samples = self.load_dvrk(samples)
        
        else:
            samples = self.load_dafoes(samples)
            samples = self.load_dvrk(samples)

        if self.mode in ["train", "val"]:
            random.shuffle(samples)
            
        self.samples = samples
    
    def load_dafoes(self, samples):

        mean_labels, std_labels = [], []
        mean_forces, std_forces = [], []

        for scene in self.scenes:
            labels = np.array(pd.read_csv(scene/'labels.csv')).astype(np.float32)
            # plot_forces(labels[140:, 0:3])
            scene_rgb = scene/"RGB_frames"
            if self.load_depths:
                scene_depth = scene/"Depth_frames"
                depth_maps = sorted(scene_depth.files("*.png"))

            #Appending mean and std for the normalization of the labels
            mean_labels.append(labels[:, :26].mean(axis=0))
            std_labels.append(labels[:, :26].std(axis=0))
            mean_forces.append((labels[:, 26:29]).mean(axis=0))
            std_forces.append((labels[:, 26:29]).std(axis=0))

            images = sorted(scene_rgb.files("*.png"))
            
            n_labels = len(labels) // len(images)
            step = 7

            for i in range(len(images)):
                if i < 20: continue
                if i + self.recurrency_size > len(images) - 20: break
                sample = {}
                sample['dataset'] = "dafoes" # Create a flag to identify it during processing
                sample['img'] = [im for im in images[i:i+self.recurrency_size]]
                if self.load_depths:
                    sample['depth'] = [depth for depth in depth_maps[i:i+self.recurrency_size]]

                sample['label'] = [np.mean(labels[n_labels*i+a: (n_labels*i+a) + step, :26], axis=0) for a in range(self.recurrency_size)]
                sample['force'] = np.mean(labels[n_labels*i+(self.recurrency_size-1):(n_labels*i+(self.recurrency_size-1)) + step, 26:29], axis=0)
                samples.append(sample)
        
        if self.mode == "train":
            self.mean_labels = np.mean(mean_labels, axis = 0) 
            self.std_labels = np.mean(std_labels, axis = 0)
            self.mean_forces = np.mean(mean_forces, axis = 0)
            self.std_forces = np.mean(std_forces, axis = 0)

            save_metric('labels_mean.npy', self.mean_labels)
            save_metric('labels_std.npy', self.std_labels)
            save_metric('forces_mean.npy', self.mean_forces)
            save_metric('forces_std.npy', self.std_forces)
        else:
            self.mean_labels, self.std_labels, self.mean_forces, self.std_forces = load_metrics("dafoes")

        return samples
    
    def load_dvrk(self, samples):

        mean_labels, std_labels = [], []
        mean_forces, std_forces = [], []

        for index in self.folder_index:

            labels, forces = read_labels(self.data_root_dvrk/'labels_{}.txt'.format(index))
            # plot_forces(forces)
            scene = self.data_root_dvrk/"imageset_{}".format(index)

            # Appending mean and std for the normalization of the labels
            mean_labels.append(labels.mean(axis=0))
            std_labels.append(labels.std(axis=0))
            mean_forces.append(forces.mean(axis=0))
            std_forces.append(forces.std(axis=0))

            images = sorted(scene.files("*.jpg"))
            labels = labels.reshape(len(images), -1)
            forces = forces.reshape(len(images), -1)


            for i in range(len(images)):
                if i < 80: continue
                if i > len(images) - (25 + self.recurrency_size): break
                sample = {}
                sample['dataset'] = "dvrk" # Flag to identify the data for processing
                sample['img'] = [scene/'img_{}.jpg'.format(i+a) for a in range(self.recurrency_size)]
                sample['label'] = [labels[i+a] for a in range(self.recurrency_size)] 
                sample['force'] = forces[i+(self.recurrency_size-1)]
                samples.append(sample)
        
        if self.mode == "train":
            self.mean_dvrk_labels = np.mean(mean_labels, axis = 0) 
            self.std_dvrk_labels = np.mean(std_labels, axis = 0)
            self.mean_dvrk_forces = np.mean(mean_forces, axis = 0)
            self.std_dvrk_forces = np.mean(std_forces, axis = 0)

            save_metric('labels_mean_dvrk.npy', self.mean_dvrk_labels)
            save_metric('labels_std_dvrk.npy', self.std_dvrk_labels)
            save_metric('forces_mean_dvrk.npy', self.mean_dvrk_forces)
            save_metric('forces_std_dvrk.npy', self.std_dvrk_forces)
        
        else:
            self.mean_dvrk_labels, self.std_dvrk_labels, self.mean_dvrk_forces, self.std_dvrk_forces = load_metrics("dvrk")

        return samples
    
    def __getitem__(self, index: int) -> Dict[str, List[torch.Tensor]]:
        sample = self.samples[index]
        imgs = [load_as_float(img) for img in sample['img']]
        
        if self.load_depths:
            if self.dataset == "dafoes":
                depths = [load_depth(depth) for depth in sample['depth']]
            elif self.dataset == "dvrk":
                depths = [np.random.randn(imgs[0].shape) for _ in range(len(imgs))]
            else:
                if sample['dataset'] == "dafoes":
                    depths = [load_depth(depth) for depth in sample['depth']]
                else:
                    depths = [np.random.randn(imgs[0].shape) for _ in range(len(imgs))]
        else:
            depths = None
        
        labels = sample['label']
        forces = sample['force']

        if self.transform is not None:
            imgs, depths, labels, forces = self.transform(imgs, depths, labels, forces, sample["dataset"])
        
        if sample['dataset'] == "dafoes":
            norm_label = np.array([(label[:26] - self.mean_labels[:26]) / (self.std_labels[:26] + 1e-10) for label in labels])
            norm_label = dafoes_2_dvrk(norm_label)
        else:
            norm_label = np.array([(label - self.mean_dvrk_labels) / (self.std_dvrk_labels + 1e-10) for label in labels])

        if self.occlude_param:
            start, end = self.occlusion[self.occlude_param]
            norm_label[:, start:end] = 0.

        if sample['dataset'] == 'dafoes':
            norm_force = (forces - self.mean_forces) / (self.std_forces + 1e-10)
        else:
            norm_force = (forces - self.mean_dvrk_forces) / (self.std_dvrk_forces + 1e-10)
        
        if self.load_depths:
            depths = [process_depth(depth) for depth in depths]
            imgd = [torch.cat([img, depth], dim=0) for img, depth in zip(imgs, depths)]
        else:
            imgd = imgs

        return {'img': imgd[0] if self.recurrency_size==1 else imgd, 'robot_state': norm_label, 
                'forces': norm_force, 'dataset': sample['dataset']}
    
    def __len__(self):
        return len(self.samples)
