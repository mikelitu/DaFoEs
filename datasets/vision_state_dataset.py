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
from datasets.augmentations import ArrayToTensor, Compose, Normalize, RandomHorizontalFlip, RandomVerticalFlip, SquareResize

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

    def __init__(self, root, recurrency_size=5, load_depths=True, max_depth=25., is_train=True, transform=None, seed=0, train_type="random"):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)

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
        self.load_depths = load_depths
        self.max_depth = max_depth
        self.recurrency_size = recurrency_size
        self.crawl_folders()
        
        
    def crawl_folders(self):
        
        samples = []
        mean_labels, std_labels = [], []
        mean_forces, std_forces = [], []

        for scene in self.scenes:
            labels = np.array(pd.read_csv(scene/'labels.csv')).astype(np.float32)
            scene_rgb = scene/"RGB_frames"
            if self.load_depths:
                scene_depth = scene/"Depth_frames"
                depth_maps = sorted(scene_depth.files("*.png"))

            #Appending mean and std for the normalization of the labels
            mean_labels.append(labels.mean(axis=0))
            std_labels.append(labels.std(axis=0))
            mean_forces.append((labels[:, -6:-3]).mean(axis=0))
            std_forces.append((labels[:, -6:-3]).std(axis=0))

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

                sample['label'] = [np.mean(labels[n_labels*i+a: (n_labels*i+a) + step], axis=0) for a in range(self.recurrency_size)]
                sample['forces'] = [np.mean(labels[n_labels*i+a:(n_labels*i+a) + step, -6:-3], axis=0) for a in range(self.recurrency_size)]
                samples.append(sample)
        
        self.mean_labels = np.mean(mean_labels, axis = 0) 
        self.std_labels = np.mean(std_labels, axis = 0)
        self.mean_forces = np.mean(mean_forces, axis = 0)
        self.std_forces = np.mean(std_forces, axis = 0)

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
        norm_force = np.array([(force - self.mean_forces) / (self.std_forces + 1e-10) for force in forces])
        
        if self.load_depths:
            depths = [process_depth(depth) for depth in depths]
            imgd = [torch.cat([img, depth], dim=0) for img, depth in zip(imgs, depths)]
        else:
            imgd = imgs

        return {'img': imgd[0] if self.recurrency_size==1 else imgd, 'robot_state': norm_label, 
                'forces': np.mean(norm_force, axis=0)}
    
    def __len__(self):
        return len(self.samples)

if __name__ == "__main__":
    root = '/home/md21local/visu_depth_haptic_data'
    normalize = Normalize(mean = [0.45, 0.45, 0.45],
                          std = [0.225, 0.225, 0.225])
    
    transforms = Compose([RandomVerticalFlip(), SquareResize(), ArrayToTensor(), normalize])
    dataset = VisionStateDataset(root, transform=transforms, recurrency_size=3)
    np.save('labels_mean.npy', dataset.mean_labels)
    np.save('labels_std.npy', dataset.std_labels)
    out = dataset[0]
    print(out['forces'])