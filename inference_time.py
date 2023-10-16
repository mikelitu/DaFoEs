import torch
from datasets.augmentations import Compose, CentreCrop, SquareResize, Normalize, ArrayToTensor
from models.force_estimator import ForceEstimator
from path import Path
from datasets.vision_state_dataset import VisionStateDataset
import numpy as np
from torch.utils.data import DataLoader
import pickle
import os
from timeit import default_timer as timer

device = torch.device("cuda")

def generate_batch(recurrency, transforms):
    # Generate an image or image list for recurrent networks
    if recurrency:
        imgs = [np.random.randn(480, 960, 3) for _ in range(5)]
        state = torch.randn((1, 5, 54))
    else:
        imgs = [np.random.randn(480, 960, 3)]
        state = torch.randn((1, 1, 54))

    imgs, _, state, _ = transforms(imgs, None, state, None)

    return imgs if recurrency else imgs[0], state.sub_(0.25).div_(0.85)



def setup(architecture, recurrency):

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

    model = ForceEstimator(architecture=architecture,
                           pretrained=False,
                           state_size=54,
                           recurrency=recurrency,
                           include_depth=False)
    
    dataset = VisionStateDataset(recurrency_size=5 if recurrency else 1,
                                 transform=transforms,
                                 load_depths=False,
                                 mode="val",
                                 dataset="chua")
    
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4)

    return model, transforms, dataloader

def run_simulation(architecture, recurrency, length=500):

    times = []
    model, transforms, dataloader = setup(architecture, recurrency)

    model.eval()
    model.to(device)

    # start = timer()
    # for i, data in enumerate(dataloader):
    for i in range(length):
        if i == length: break
        img, state = generate_batch(recurrency, transforms)
        start = timer()
        img = [im.to(device).unsqueeze(0) for im in img] if recurrency else img.to(device).unsqueeze(0)
        state = state.to(device) if recurrency else state.squeeze(1).to(device)
        # img = [im.to(device) for im in data["img"]] if recurrency else data["img"].to(device)
        # state = data["robot_state"].squeeze(1).to(device)
        _ = model(state) if architecture=="fc" else model(img, state)
        end = timer()
        times.append(end - start)
        # start = timer()
    
    return np.mean(times)

def main():
    architectures = ["cnn", "vit", "fc", "rcnn", "rvit"]

    print("Avoiding problems with the loader and some previous effects.")

    for arch in architectures:
        pr_arch = arch
        print("Starting experiment for {}...".format(arch))
        if arch in ["rcnn", "rvit"]:
            recurrency = True
            if arch == "rcnn": arch = "cnn"
            else: arch = "vit"
        else:
            recurrency = False
        
        inference_time = run_simulation(arch, recurrency, 1000)
        print("The mean inference time for model {} was: {} s".format(pr_arch, inference_time))
        print("Therefore the mean frequency is: {} Hz".format(1 / inference_time))

if __name__ == "__main__":
    main()

