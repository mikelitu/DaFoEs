import torch
from models.force_estimator import ForceEstimator
from torch.utils.data import DataLoader
from datasets.vision_state_dataset import VisionStateDataset
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
parser.add_argument("--dataset", default="mixed", type=str, choices=["img2force", "chua", "mixed"],
                    help="The dataset loading for the experiment")
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
    data = args.dataset
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

    dataset = VisionStateDataset(recurrency_size=recurrency_size, transform=transforms, dataset="mixed", load_depths=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = ForceEstimator(architecture=architecture, state_size=54, recurrency=recurrency, pretrained=False, include_depth=False)
    checkpoint_path = Path('{}/checkpoints/{}/rgb/{}/{}_random'.format(data_root, data, "r"+architecture if recurrency else architecture, "visu_state" if include_state else "visu"))
    checkpoint_dir = sorted(checkpoint_path.dirs())[-1]/'checkpoint.pth.tar'
    print("The checkpoints are loaded from: {}".format(checkpoint_dir))

    checkpoint = torch.load(checkpoint_dir)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.to(device)
    model.eval()

    heatmap_frames_img2force = []
    heatmap_frames_chua = []


    for i, data in enumerate(tqdm(dataloader)):
        if i < 20: continue

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
        heatmaps = [np.uint8(255 * heatmap) for heatmap in heatmaps]
        heatmaps = [cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) for heatmap in heatmaps]
        bgr_img = cv2.cvtColor(plt_img, cv2.COLOR_RGB2BGR)
        bgr_img *= 255 / bgr_img.max()

        superimposed_img_x = bgr_img.astype(np.float32) + 0.4 * heatmaps[0].astype(np.float32)
        superimposed_img_x *= 255 / superimposed_img_x.max()
        superimposed_img_x = np.uint8(superimposed_img_x)
        superimposed_img_x = cv2.putText(superimposed_img_x, "GT: {:.2f}N | Pred: {:.2f}N".format(force[:, 0].item(), pred[:, 0].item()),
                                         (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=2)

        superimposed_img_y = bgr_img.astype(np.float32) + 0.4 * heatmaps[1].astype(np.float32)
        superimposed_img_y *= 255 / superimposed_img_y.max()
        superimposed_img_y = np.uint8(superimposed_img_y)
        superimposed_img_y = cv2.putText(superimposed_img_y.astype(np.uint8), "GT: {:.2f}N | Pred: {:.2f}N".format(force[:, 1].item(), pred[:, 1].item()),
                                         (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=2)
        

        superimposed_img_z = bgr_img.astype(np.float32) + 0.4 * heatmaps[2].astype(np.float32)
        superimposed_img_z *= 255 / superimposed_img_z.max()
        superimposed_img_z = np.uint8(superimposed_img_z)
        superimposed_img_z = cv2.putText(superimposed_img_z, "GT: {:.2f}N | Pred: {:.2f}N".format(force[:, 2].item(), pred[:, 2].item()),
                                         (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=2)
        
        superimposed_img = np.concatenate([superimposed_img_x, superimposed_img_y, superimposed_img_z], axis=1)

        if data["dataset"] == "img2force":
            heatmap_frames_img2force.append(superimposed_img)
        else:
            heatmap_frames_chua.append(superimposed_img)

    print("Generating video...")

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out_heatmaps_img2force = cv2.VideoWriter(saving_dir/'grad_cam_img2force.mp4', fourcc, 20., (3 * 256, 256))
    
    for h in tqdm(heatmap_frames_img2force):
        out_heatmaps_img2force.write(h)
    out_heatmaps_img2force.release()
    print("Videos saved at: {}/grad_cam_img2force.mp4".format(saving_dir))

    out_heatmaps_chua = cv2.VideoWriter(saving_dir/'grad_cam_chua.mp4', fourcc, 20., (3 * 256, 256))
    
    for h in tqdm(heatmap_frames_chua):
        out_heatmaps_chua.write(h)
    out_heatmaps_chua.release()
    print("Videos saved at: {}/grad_cam_chua.mp4".format(saving_dir))
    

if __name__ == "__main__":
    main()