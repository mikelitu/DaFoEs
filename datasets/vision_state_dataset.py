import torch
import torch.nn as nn
import os
from typing import Dict, List

from torch.utils.data import Dataset, DataLoader
import imageio
import numpy as np
import random
from path import Path
import pandas as pd
from PIL import ImageFile, Image
from datasets.utils import RGBtoD
from datasets.augmentations import BrightnessContrast, Compose
import matplotlib.pyplot as plt
from datasets.utils import plot_forces, save_metric

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_as_float(path: Path) -> np.ndarray:
    return  imageio.imread(path)[:,:, :3].astype(np.float32)

def load_depth(path: Path) -> np.ndarray:
    return np.array(Image.open(path)).astype(np.uint16).astype(np.float32)

def process_depth(rgb_depth: torch.Tensor) -> torch.Tensor:
    depth = torch.zeros((1, rgb_depth.shape[1], rgb_depth.shape[2]))
    for i in range(rgb_depth.shape[1]):
        for j in range(rgb_depth.shape[2]):
            depth[:, i, j] = RGBtoD(rgb_depth[0, i, j], rgb_depth[1, i, j], rgb_depth[2, i, j])
    
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

    def __init__(self, root, recurrency_size=5, load_depths=True, max_depth=25., is_train=True, transform=None, seed=0, train_type="random",
                 occlude_param=None):
        
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)

        self.occlusion = {"robot_p": [0, 7],
                          "robot_j": [7, 13],
                          "haptics_p": [13, 20],
                          "haptics_j": [20, 26]}
        
        train_files = {"random": "train.txt", 
                    "geometry": "train_geometry.txt", 
                    "color": "train_color.txt", 
                    "structure": "train_structure.txt",
                    "stiffness": "train_stiffness.txt",
                    "position": "train_position.txt"}

        val_files = {"random": "val.txt", 
                    "geometry": "val_geometry.txt", 
                    "color": "val_color.txt", 
                    "structure": "val_structure.txt",
                    "stiffness": "val_stiffness.txt",
                    "position": "val_position.txt"}

        scene_list_path = self.root/train_files[train_type] if is_train else self.root/val_files[train_type]
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)][:-1]
        self.transform = transform
        self.is_train = is_train
        self.load_depths = load_depths
        self.max_depth = max_depth
        self.recurrency_size = recurrency_size
        self.occlude_param = occlude_param
        self.crawl_folders()
        
        
    def crawl_folders(self):
        
        samples = []
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
                sample['img'] = [im for im in images[i:i+self.recurrency_size]]
                if self.load_depths:
                    sample['depth'] = [depth for depth in depth_maps[i:i+self.recurrency_size]]

                sample['label'] = [np.mean(labels[n_labels*i+a: (n_labels*i+a) + step, :26], axis=0) for a in range(self.recurrency_size)]
                sample['forces'] = np.mean(labels[n_labels*i+(self.recurrency_size-1):(n_labels*i+(self.recurrency_size-1)) + step, 26:29], axis=0)
                samples.append(sample)
        
        self.mean_labels = np.mean(mean_labels, axis = 0) 
        self.std_labels = np.mean(std_labels, axis = 0)
        self.mean_forces = np.mean(mean_forces, axis = 0)
        self.std_forces = np.mean(std_forces, axis = 0)

        save_metric('labels_mean.npy', self.mean_labels)
        save_metric('labels_std.npy', self.std_labels)
        save_metric('forces_mean.npy', self.mean_forces)
        save_metric('forces_std.npy', self.std_forces)

        random.shuffle(samples)
        self.samples = samples
    
    def __getitem__(self, index: int) -> Dict[str, List[torch.Tensor]]:
        sample = self.samples[index]
        imgs = [load_as_float(img) for img in sample['img']]
        if self.load_depths:
            depths = [load_depth(depth) for depth in sample['depth']]
        else:
            depths = None
        
        labels = sample['label']
        forces = sample['forces']

        if self.transform is not None:
            imgs, depths, labels, forces = self.transform(imgs, depths, labels, forces)
        
        norm_label = np.array([(label[:26] - self.mean_labels[:26]) / (self.std_labels[:26] + 1e-10) for label in labels])

        if self.occlude_param is not None and self.is_train:
            start, end = self.occlusion[self.occlude_param]
            norm_label[:, start:end] = 0.

        norm_force = (forces - self.mean_forces) / (self.std_forces + 1e-10)
        
        if self.load_depths:
            depths = [process_depth(depth) for depth in depths]
            imgd = [torch.cat([img, depth], dim=0) for img, depth in zip(imgs, depths)]
        else:
            imgd = imgs

        return {'img': imgd[0] if self.recurrency_size==1 else imgd, 'robot_state': norm_label, 
                'forces': norm_force}
    
    def __len__(self):
        return len(self.samples)

if __name__ == "__main__":
    root = '/home/md21local/visu_depth_haptic_data'
    
    brightcont = BrightnessContrast(contrast=2., brightness=12.)
    transforms = Compose([brightcont])
    dataset = VisionStateDataset(root, transform=transforms, recurrency_size=1, load_depths=False, occlude_param="haptics_p")
    data = dataset[10]

    print(data['forces'])
    