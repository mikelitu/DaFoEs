import torch
from torch.utils.data import Dataset
import imageio
import random
from path import Path
from PIL import ImageFile
import numpy as np
import pandas as pd
from datasets.chua_dataset import read_labels
from datasets.utils import load_metrics
from datasets.vision_state_dataset import load_depth, process_depth

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_as_float(path: Path) -> np.ndarray:
    return  imageio.imread(path).astype(np.float32)


class TestDataset(Dataset):

    def __init__(self, root_dir: Path, transform = None, recurrency_size: int = 5, dataset: str = "img2force", load_depths: bool = False) -> None:
        super(TestDataset, self).__init__()

        self.root = root_dir

        scene_list_path = self.root/"test.txt"
        
        if dataset == "chua":
            self.folder_index = [folder.split('_')[-1].rstrip('\n') for folder in open(scene_list_path)]
        else:
            self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)][:-1]

        self.transform = transform
        self.load_depths = load_depths
        self.recurrency_size = recurrency_size
        self.mean_labels, self.std_labels, self.mean_forces, self.std_forces = load_metrics(dataset)
        self.dataset = dataset
        self.crawl_folders()

    def crawl_folders(self):
        samples = []

        if self.dataset == "img2force":
            for scene in self.scenes:
                labels = np.array(pd.read_csv(scene/'labels.csv')).astype(np.float32)
                scene_rgb = scene/"RGB_frames"
                images = sorted(scene_rgb.files("*.png"))
                if self.load_depths:
                    scene_depth = scene/"Depth_frames"
                    depth_maps = sorted(scene_depth.files("*.png"))

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
        else:   
            for index in self.folder_index:
                labels, forces = read_labels(self.root/'labels_{}')
                scene = self.root/"imageset_{}".format(index)
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
        
        self.samples = samples
    
    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        imgs = [load_as_float(img) for img in sample['img']]
        
        if self.load_depths:
            depths = [load_depth(depth) for depth in sample['depth']]
        else:
            depths = None
        
        labels = sample['label']
        forces = sample['forces']

        if self.transform is not None:
            imgs, depths, labels, forces = self.transform(imgs, depths, labels, forces, self.dataset)
        
        norm_label = np.array([(label - self.mean_labels)/ (self.std_labels + 1e-10) for label in labels])
        norm_force = (forces - self.mean_forces) / (self.std_forces + 1e-10)

        if self.load_depths:
            depths = [process_depth(depth) for depth in depths]
            imgd = [torch.cat([img, depth], dim=0) for img, depth in zip(imgs, depths)]
        else:
            imgd = imgs

        return {'img': imgd, 'robot_state': np.mean(norm_label, axis=0).astype(np.float32) if self.recurrency_size==1 else np.array(norm_label).astype(np.float32),
                'forces': norm_force}

    def __len__(self):
        return len(self.samples)