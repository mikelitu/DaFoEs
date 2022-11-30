import models.utils as utils
import torch
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from models.force_estimator_2d import ForceEstimatorV, ForceEstimatorVS, ForceEstimatorS
import torch.nn as nn


def main():
    
    model_vs = ForceEstimatorVS(rs_size=54)
    model_v = ForceEstimatorV()
    model_s = ForceEstimatorS(rs_size=54)

    image = torch.randn((2, 3, 256, 256))
    robot_state = torch.randn((2, 54))
    
    out_v = model_v(image)
    out_s = model_s(robot_state)
    out_vs = model_vs(image, robot_state)

    print(out_s)
    print(out_v)
    print(out_vs)

    lstm =nn.LSTM(input_size=512, hidden_size=500, num_layers=2, batch_first=True, dropout=0.1)
    print(lstm)

if __name__ == "__main__":
    main()

