import torch
import torch.nn as nn
import os

from torch.utils.data import Dataset, DataLoader
import imageio
import numpy as np
import random
from path import Path
import pandas as pd
from PIL import ImageFile
from datasets.utils import plot_forces, save_metric
import datasets.augmentations as augmentations

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

def load_as_float(path: Path) -> np.ndarray:
    return  imageio.imread(path).astype(np.float32)



class ZhongeChuaDataset(Dataset):
    """A dataset to load data from dfferent folders that are arranged this way:
        root/scene_1/000.png
        root/scene_1/001.png
        ...
        root/scene_1/labels.csv
        root/scene_2/000.png
        .
        transform functions takes in a list images and a numpy array representing the intrinsics of the camera and the robot state
    """

    def __init__(self, root, is_train=True, recurrency_size=5, transform=None, seed=0, train_type="random"):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)

        scene_list_path = self.root/"train.txt" if is_train else self.root/"val.txt"
        self.folder_index = [folder.split('_')[-1].rstrip('\n') for folder in open(scene_list_path)]
        # self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.transform = transform
        self.recurrency_size = recurrency_size
        self.crawl_folders()
        
        
    def crawl_folders(self):
        
        samples = []
        mean_labels, std_labels = [], []
        mean_forces, std_forces = [], []

        for index in self.folder_index:

            labels, forces = read_labels(self.root/'labels_{}.txt'.format(index))
            # plot_forces(forces)
            scene = self.root/"imageset_{}".format(index)

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
                sample['img'] = [scene/'img_{}.jpg'.format(i+a) for a in range(self.recurrency_size)]
                sample['state'] = [labels[i+a] for a in range(self.recurrency_size)] 
                sample['force'] = forces[i+(self.recurrency_size-1)]
                samples.append(sample)
        
        self.mean_labels = np.mean(mean_labels, axis = 0) 
        self.std_labels = np.mean(std_labels, axis = 0)
        self.mean_forces = np.mean(mean_forces, axis = 0)
        self.std_forces = np.mean(std_forces, axis = 0)

        save_metric('labels_mean_chua.npy', self.mean_labels)
        save_metric('labels_std_chua.npy', self.std_labels)
        save_metric('forces_mean_chua.npy', self.mean_forces)
        save_metric('forces_std_chua.npy', self.std_forces)

        random.shuffle(samples)
        self.samples = samples
    
    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        img = [load_as_float(img) for img in sample['img']]
        state = np.array([(s - self.mean_labels) / (self.std_labels + 1e-10) for s in sample['state']])
        force = np.array((sample['force'] - self.mean_forces) / (self.std_forces + 1e-10))

        if self.transform is not None:
            img, _, state, force = self.transform(img, depths=None, states=state, forces=force, model="chua")
        
        if self.recurrency_size == 1:
            img = img[0]
        
        return {'img': img, 'robot_state': np.mean(state, axis=0).astype(np.float32) if self.recurrency_size==1 else np.array(state).astype(np.float32), 
                'forces': force}
    
    def __len__(self):
        return len(self.samples)

if __name__ == "__main__":
    root_dir = Path("/home/md21local/experiment_data")
    transform = augmentations.Compose([augmentations.RandomHorizontalFlip(), 
                                       augmentations.RandomVerticalFlip(),
                                       augmentations.RandomRotation()
                                       ])
    
    dataset = ZhongeChuaDataset(root_dir, recurrency_size=5, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1)

    for i, data in enumerate(dataloader):
        print(i)
        continue