import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from path import Path
import random
from datasets.utils import check_key, check_params, generate_keys_rs


class StateDataset(Dataset):

    def __init__(self, root, is_train=True, seed=0, occlude_params=[]):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        self.keys = generate_keys_rs()
        self.occluded_params = check_params(self.keys, occlude_params)
        _ = self.occ()
        scene_list_path = self.root/"train.txt" if is_train else self.root/"val.txt"
        self.scenes = [self.root/folder[-1] for folder in open(scene_list_path)]
        self.scrap_labels()


    def scrap_labels(self):
        samples = []
        for scene in self.scenes:
            labels = np.genfromtxt(scene/"labels.txt").astype(np.float32).reshape(-1, "number to be decided")
            states = dict()

            for label in enumerate(labels):
                #Here we need to decide what are the params we want to include from the robot state
                #There are some parameters that will be certainly included
                states = self.create_occluded_dict(label)
                samples.append(states)
        
        random.shuffle(samples)
        self.samples = samples

    def create_occluded_dict(self, label):
        #This code assumes you know the order from your label data list and information (keys) it contains at all of them are predifined
        #Create an empty dictionary
        dic = dict()
        #We iterate over each key to see if they are occluded or not
        for key in self.keys:
            range = check_key(key)
            value = label[range[0]:range[1]] 
            if key in self.occlude_params:
                dic[key] = np.zeros_like(value)
            else:
                dic[key] = value
        
        return dic
    
    def occ(self):
        print("The parameters occluded for this dataset are: ", self.occluded_params)
        return self.occluded_params
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return sample
            
            
