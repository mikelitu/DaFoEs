import torch
from models.recorder import BamRecorderV, BamRecorderVS, SampRecorder, Recorder
from models.force_estimator_2d import ForceEstimatorV, ForceEstimatorVS
from models.force_estimator_transformers import ViT
from models.force_estimator_transformers_base import BaseViT
from datasets.augmentations import CentreCrop, SquareResize, Normalize, ArrayToTensor, Compose, GaussianNoise
from datasets.vision_state_dataset import normalize_labels, load_as_float
import seaborn as sns
import matplotlib.pyplot as plt
from path import Path
import cv2
import imageio
import numpy as np
import pandas as pd


num_img = 307
feature = "random"
include_state = False
model_name = 'vit'

# The different possible models to analyse the attention map

models = {
    'vit': ViT(
    image_size = 256,
    patch_size = 16,
    num_classes = 6,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1,
    max_tokens_per_depth=(256, 128, 64, 32, 16, 8),
    state_include = include_state
    ),

    'vit-base': BaseViT(
        image_size = 256,
        patch_size = 16,
        num_classes = 6,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1,
        state_include = include_state
    ),

    'vit-dist': BaseViT(
        image_size = 256,
        patch_size = 16,
        num_classes = 6,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1,
        state_include = include_state
    ),


    'cnn-bam': 
    ForceEstimatorVS(num_layers=50, pretrained=False, att_type='BAM', rs_size=25) if include_state else ForceEstimatorV(num_layers=50, pretrained=False, att_type='BAM')
}

#Recorders

recorders = {'vit': SampRecorder, 'vit-base': Recorder, 'vit-dist': Recorder, 'cnn-bam': BamRecorderVS if include_state else BamRecorderV}

def plot_attention(attns: torch.Tensor):

    # attns -> (batch x layers x heads x patch x patch)
    if not isinstance(attns, list):
        attns = attns.squeeze(0).cpu().numpy()
    
    
    for i, attention in enumerate(attns):
        fig = plt.figure(figsize=(1920/100, 1080/100), dpi=100)
        ax = sns.heatmap(attention[:, 1:, 1:].mean(axis=0).tolist(),
                        vmin=attention[:, 1:, 1:].mean(axis=0).min(),
                        vmax=attention[:, 1:, 1:].mean(axis=0).max(),
                        cmap='Reds', cbar=False)
        plt.axis("off")

        plt.show()


def main():

    
    img_file = '/home/md21local/visu_haptic_data/E_P_S_P_C/{}.png'.format(str(num_img).zfill(4))
    img = imageio.imread(img_file).astype(np.float32)
    if include_state:
        labels = np.array(pd.read_csv('/home/md21local/visu_haptic_data/E_P_S_P_C/labels.csv')).astype(np.float32)
        nlabels = len(labels) // 1467
        norm_labels = normalize_labels(labels)
        state = torch.from_numpy(norm_labels[nlabels * num_img, :-6]).unsqueeze(0)

    normalize = Normalize(
        mean = [0.45, 0.45, 0.45],
        std = [0.225, 0.225, 0.225]
    )


    transforms = Compose([
        CentreCrop(),
        SquareResize(),
        ArrayToTensor(),
        normalize,
    ])

    img = transforms([img])[0]

    imageio.imsave('test.png', img.permute(1, 2, 0).numpy().astype(np.uint8))
    model = models[model_name]

    model.eval()
    root_checkpoint_dir = Path('/home/md21local/mreyzabal/checkpoints/old_img2force/{}/{}_{}'.format(model_name, "visu_state" if include_state else "visu", feature))
    checkpoint_dir = root_checkpoint_dir.dirs()[-1]
    checkpoints = torch.load(checkpoint_dir/'checkpoint.pth.tar')
    model.load_state_dict(checkpoints['state_dict'], strict=True)

    model = recorders[model_name](model)

    if include_state:
        if model_name == 'cnn-bam':
            _, attns = model(img.unsqueeze(0), state)
        else:
            _, attns = model(img.unsqueeze(0), None, state)

    else:
        _, attns = model(img.unsqueeze(0))
    
    if isinstance(attns, list):
        attns = [att.squeeze(0).detach().numpy() for att in attns]
    plot_attention(attns)


if __name__ == "__main__":
    main()