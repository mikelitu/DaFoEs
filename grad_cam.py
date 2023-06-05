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

def reshape_transform(tensor: torch.Tensor, height: int = 16, width: int = 16):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                height, width, tensor.size(2))
    
    # Bring the channels to the first dimension like CNNs
    result = result.transpose(2, 3).transpose(1, 2)

    return result

architecture = "cnn"

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

root_dir = Path('/home/md21local/experiment_data')
dataset = TestDataset(root_dir=root_dir, recurrency_size=1, transform=transforms, dataset="chua")
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)


# plt.imshow(plt_img)
# plt.show()

model = ForceEstimator(architecture=architecture, state_size=54, recurrency=False, pretrained=False, include_depth=False)
checkpoint_path = Path('/home/md21local/mreyzabal/checkpoints/chua/rgb/{}/visu_state_random/05-19-17:52/checkpoint.pth.tar'.format(architecture))
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['state_dict'], strict=False)
model.eval()

for i, data in enumerate(dataloader):
    plt_img = inv_transform(data['img'])[0].squeeze().permute(1, 2, 0).numpy()
    pred = model(data['img'], data['robot_state'])
    heatmaps = []

    for a in range(3):
        pred[:, a].backward(retain_graph=True)

        gradients = model.get_activations_gradient()

        if architecture == "vit":
            gradients = reshape_transform(gradients)
        
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        activations = model.get_activations(data['img']).detach()
        
        if architecture == "vit":
            activations = reshape_transform(activations)

        for i in range(pooled_gradients.shape[0]):
            activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze()

        for x in range(heatmap.shape[0]):
            for y in range(heatmap.shape[1]):
                heatmap[x, y] = max(heatmap[x, y].item(), 0)

        heatmap = np.maximum(heatmap, 0)
        heatmap /= torch.max(heatmap)
        heatmaps.append(heatmap)
        # plt.matshow(heatmap.squeeze())
        # plt.show()

    heatmaps = [cv2.resize(heatmap.numpy(), (plt_img.shape[0], plt_img.shape[1])) for heatmap in heatmaps]
    heatmaps = [np.uint8(255 * heatmap) for heatmap in heatmaps]
    heatmaps = [cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) for heatmap in heatmaps]
    # plt.imshow(heatmap, vmax=heatmap.max(), vmin=heatmap.min())
    # plt.show()

    plt.subplot(1,3,1)
    plt.imshow(plt_img)
    plt.imshow(heatmaps[0], alpha=0.3)
    plt.title("Grad-CAM X")
    plt.axis("off")
    plt.subplot(1,3,2)
    plt.imshow(plt_img)
    plt.imshow(heatmaps[1], alpha=0.3)
    plt.title("Grad-CAM Y")
    plt.axis("off")
    plt.subplot(1,3,3)
    plt.imshow(plt_img)
    plt.imshow(heatmaps[2], alpha=0.3)
    plt.title("Grad-CAM Z")
    plt.axis("off")
    plt.pause(0.001)
    # cv2.imwrite('figures/grad_cam.png', superimposed_img)
    # plt.imshow(superimposed_img)
    # plt.show()

