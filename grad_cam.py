import torch
from models.force_estimator import ForceEstimator
from torch.utils.data import DataLoader
from datasets.vision_state_dataset import VisionStateDataset
from datasets.test_dataset import TestDataset
from datasets.augmentations import Normalize, BrightnessContrast, SquareResize, CentreCrop, Compose, ArrayToTensor
from path import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import os
import imageio
from tqdm import tqdm

torch.backends.cudnn.enabled=False

root = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(description="Algorithm for grad-cam on the different train architectures",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--save-dir', default='grads', type=str,
                    help='Saving directory for the superimposed gradients')
parser.add_argument('--type', default='vs', choices=['v', 'vs'], type=str,
                    help='define if we have a multimodal network or not')
parser.add_argument('--architecture', choices=['cnn', 'vit'], type=str,
                    help='the base architecture of the chosen network')
parser.add_argument('--chua', action='store_true', help="decide if the used dataset is the one from Chua's paper")
parser.add_argument('--recurrency', action='store_true', help="add recurrent blocks as the last processing layer")
parser.add_argument('--cuda', action='store_true', help='flag to activate the use of GPU acceleration')


def reshape_transform(tensor: torch.Tensor, height: int = 16, width: int = 16):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                height, width, tensor.size(2))
    
    # Bring the channels to the first dimension like CNNs
    result = result.transpose(2, 3).transpose(1, 2)

    return result


def grad_cam(model: ForceEstimator, img: torch.Tensor, architecture: str):
    
    gradients = model.get_activations_gradient()

    if architecture == "vit":
        gradients = reshape_transform(gradients)
    
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    activations = model.get_activations(img).detach()
    
    if architecture == "vit":
        activations = reshape_transform(activations)

    for i in range(pooled_gradients.shape[0]):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze()

    for x in range(heatmap.shape[0]):
        for y in range(heatmap.shape[1]):
            heatmap[x, y] = max(heatmap[x, y].item(), 0)

    heatmap = np.maximum(heatmap.cpu(), 0)
    heatmap /= torch.max(heatmap)
    return heatmap

def main():
    args = parser.parse_args()
    architecture = args.architecture
    data = "chua" if args.chua else "img2force"
    include_state = True if args.type=="vs" else False
    recurrency = args.recurrency
    recurrency_size = 5 if recurrency else 1
    device = torch.device("cuda" if args.cuda else "cpu") 

    saving_dir = Path("{}/{}/{}/{}/{}".format(root, args.save_dir, data, "r"+architecture if recurrency else architecture, "visu_state" if include_state else "visu"))
    saving_dir.makedirs_p()

    normalize = Normalize(mean=[0.45, 0.45, 0.45],
                        std=[0.225, 0.225, 0.225])

    inv_normalize = Normalize(mean=[-0.45/0.225, -0.45/0.225, -0.45/0.225],
                        std=[1/0.225, 1/0.225, 1/0.225])

    # brightness = BrightnessContrast(contrast=2., brightness=12.)

    transforms = Compose([CentreCrop(),
                        SquareResize(),
                        ArrayToTensor(),
                        normalize
                        ])

    inv_transform = Compose([inv_normalize])

    root_dir = Path("{}/{}".format(data_root, "experiment_data" if args.chua else "visu_depth_haptic_data"))
    dataset = TestDataset(root_dir=root_dir, recurrency_size=recurrency_size, transform=transforms, dataset=data)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = ForceEstimator(architecture=architecture, state_size=54 if args.chua else 26, recurrency=recurrency, pretrained=False, include_depth=False)
    checkpoint_path = Path('{}/checkpoints/{}/rgb/{}/{}_random'.format(data_root, data, "r"+architecture if recurrency else architecture, "visu_state" if include_state else "visu"))
    checkpoint_dir = sorted(checkpoint_path.dirs())[-1]/'checkpoint.pth.tar'
    print("The checkpoints are loaded from: {}".format(checkpoint_dir))

    checkpoint = torch.load(checkpoint_dir)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.to(device)
    model.eval()

    video_frames = []

    for i, data in enumerate(tqdm(dataloader)):
        img = [im.to(device) for im in data['img']] if recurrency else data['img'].to(device)
        if include_state:
            state = data['robot_state'].to(device)
        else:
            state = None
        force = data['forces']

        plt_img = inv_transform(data['img'])[0][4].squeeze().permute(1, 2, 0).numpy() if recurrency else inv_transform(data['img'])[0].squeeze().permute(1, 2, 0).numpy()
        pred = model(img, state)

        heatmaps = []

        for a in range(3):
            pred[:, a].backward(retain_graph=True)
            heatmap = grad_cam(model, img, architecture)
            heatmaps.append(heatmap)

        heatmaps = [cv2.resize(heatmap.numpy(), (plt_img.shape[0], plt_img.shape[1])) for heatmap in heatmaps]
        heatmaps = [np.clip(heatmap, 0, 1) for heatmap in heatmaps]
        heatmaps = [np.uint8(255. * heatmap) for heatmap in heatmaps]
        heatmaps = [cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) for heatmap in heatmaps]
        plt_img = cv2.cvtColor(plt_img, cv2.COLOR_RGB2BGR)

        superimposed_img_x = plt_img + 0.3 * heatmaps[0]
        superimposed_img_x = cv2.putText(superimposed_img_x, "GT: {:.2f}N | Pred: {:.2f}N".format(force[:, 0].item(), pred[:, 0].item()),
                                         (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=2)

        superimposed_img_y = plt_img + 0.3 * heatmaps[1]
        superimposed_img_y = cv2.putText(superimposed_img_y, "GT: {:.2f}N | Pred: {:.2f}N".format(force[:, 1].item(), pred[:, 1].item()),
                                         (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=2)
        
        superimposed_img_z = plt_img + 0.3 * heatmaps[2]
        superimposed_img_z = cv2.putText(superimposed_img_z, "GT: {:.2f}N | Pred: {:.2f}N".format(force[:, 2].item(), pred[:, 2].item()),
                                         (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=2)
        
        superimposed_img = np.concatenate([superimposed_img_x, superimposed_img_y, superimposed_img_z], axis=1)

        video_frames.append(superimposed_img)
    
    print("Generating video...")
    # with imageio.get_writer(saving_dir/"grad_cam.mp4", fps=20) as writer:
    #     for f in tqdm(video_frames):
    #         writer.append_data(f)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(saving_dir/'grad_cam.mp4', fourcc, 20, (256, 256), isColor=True)

    for f in tqdm(video_frames):
        out.write(f)
    
    out.release()
    

if __name__ == "__main__":
    main()