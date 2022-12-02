import models.utils as utils
import torch
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from models.force_estimator_2d import ForceEstimatorV, ForceEstimatorVS, ForceEstimatorS, ResNet18
import torch.nn as nn


def main():
    
    dic = {'robot_pos': [], 'joint_pos': [], 'com_pos': []}
    occlude = ['robot_pos', 'joint_pos']

    keys = list(dic.keys())
    
    for occ in occlude:
        if occ in keys:
            print("you have succesfully occluded {}".format(occ))


if __name__ == "__main__":
    main()



