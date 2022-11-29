import models.utils as utils
import torch
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from models.force_estimator import ResNet18


def main():
    
    model = ResNet18(in_channels=3)
    print(model)

    image = torch.randn((1, 3, 256, 256))
    out = model(image)
    print(out.shape)

if __name__ == "__main__":
    main()

