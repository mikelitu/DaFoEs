import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from torch.utils.data import DataLoader
from losses import RMSE, EuclideanDist
import argparse
from models.force_estimator_2d import ForceEstimatorS, ForceEstimatorV, ForceEstimatorVS, RecurrentCNN
from models.force_estimator_3d import KPDetector
import datetime
import time
import csv
import numpy as np
from datasets import augmentations
from utils import save_checkpoint
from datasets.base_dataset import VisionStateDataset
from datasets.state_dataset import StateDataset
from tensorboardX import SummaryWriter
from path import Path

parser = argparse.ArgumentParser(description='Vision and roboto state based force estimator using CNNs',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--model-type', type=str, choices=['2d_v', '2d_vs', '2d_s', '2d_rnn', '3d_conv'], default='3d_conv', help='the chosen model type')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M', help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--seed', default=0, type=int, help='seed for random function, and network initialization')
parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH', help='csv to save the per-epoch train and valid stats')
parser.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH', help='csv to save the progress per gradient during training')
parser.add_argument('--log-output', action='store_true', help='will log force estimation ouputs at validation')
parser.add_argument('--resnet-layers', type=int, default=18, choices=[18, ], help='number of ResNet layers')
parser.add_argument('--pretrained-model', default=None, metavar='PATH', help='path to pretrained model')
parser.add_argument('--name', dest='name', type=str, required=True, help='name of the experiment, checkpoint are stored in checkpoint/name')

best_error = -1
n_iter = 0

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.autograd.set_detect_anomaly(True) 

def main():
    global best_error, n_iter, device
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    save_path = Path(args.name)
    args.save_path = 'checkpoints'/save_path/timestamp
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


    ##Initialize the optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    if ckp is not None:
        load_checkpoint(ckp, model, optimizer)
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)

    #Start the losses
    rmse = RMSE()
    euc_dist = EuclideanDist()

    
    for epoch in tqdm(range(n_epochs)):
        for (i, data) in tqdm(enumerate(dataloader)):
            #Isolate the data into the different parts and save them properly
            continue
