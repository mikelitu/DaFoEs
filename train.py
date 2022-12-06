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
from utils import save_checkpoint, assert_dataset_type, assert_model_type
from datasets.vision_state_dataset import VisionStateDataset
from datasets.state_dataset import StateDataset
from tensorboardX import SummaryWriter
from path import Path
from logger import TermLogger, AverageMeter

parser = argparse.ArgumentParser(description='Vision and roboto state based force estimator using CNNs',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--model-type', type=str, choices=['2d_v', '2d_vs', '2d_s', '2d_rnn', '3d_cnn'], default='3d_conv', help='the chosen model type')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M', help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--seed', default=0, type=int, help='seed for random function, and network initialization')
parser.add_argument('--log-summary', default='log_summary.csv', metavar='PATH', help='csv to save the per-epoch train and valid stats')
parser.add_argument('--log-full', default='log_full.csv', metavar='PATH', help='csv to save the progress per gradient during training')
parser.add_argument('--log-output', action='store_true', help='will log force estimation ouputs at validation')
parser.add_argument('--resnet-layers', type=int, default=18, choices=[18, ], help='number of ResNet layers')
parser.add_argument('--pretrained', default=None, metavar='PATH', help='path to pretrained model')
parser.add_argument('--name', dest='name', type=str, required=True, help='name of the experiment, checkpoint are stored in checkpoint/name')
parser.add_argument('--occ', default=None, choices=['robot_pos', 'joint_pos', 'geo_com'], nargs='+')
parser.add_argument('--verbose', dest='verbose', type=bool, action="store_true", help="print model architecture")

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
    cudnn.deterministic = True
    cudnn.benchmark = True

    training_writer = SummaryWriter(args.save_path)
    output_writers = []
    if args.log_output:
        for i in range(3):
            output_writers.append(SummaryWriter(args.save_path/'valid'/str(i)))


    #Initialize the transformations
    normalize = augmentations.Normalize(mean = [0.45, 0.45, 0.45],
                                        std = [0.225, 0.225, 0.225])
    resize = augmentations.Resize()
    jitter = augmentations.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)

    train_transform = augmentations.Compose([
        augmentations.RandomHorizontalFlip(),
        augmentations.RandomScaleCrop(),
        resize,
        jitter,
        augmentations.ArrayToTensor(),
        normalize
    ])

    val_transform = augmentations.Compose([
        augmentations.RandomScaleCrop(),
        resize,
        augmentations.ArrayToTensor(),
        normalize
    ])
    
    print("=> Getting scenes from '{}'".format(args.data))
    print("=> Choosing the correct dataset for choice {}...".format(args.type))
    
    train_dataset, val_dataset = assert_dataset_type(args, train_transform, val_transform)

    print('{} samples found in {} train scenes'.format(len(train_dataset), len(train_dataset.scenes)))
    print('{} samples found in {} validation scenes'.format(len(val_dataset), len(val_dataset.scenes)))

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True
    )

    # Create the model
    print("=> Creating the correct model for choice {}...".format(args.type))
    model, model_name = assert_model_type(args)
    print("The chosen model is {}".format(model_name))
    if args.verbose:
        print(model)

    #Load parameters
    if args.pretrained:
        print("=> Using pre-trained weights for {}".format(model_name))
        weights = torch.load(args.pretrained)
        model.load_state_dict(weights['state_dict'], strict=False)
    
    print("=> Setting Adam optimizer")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.momentum, args.beta), weight_decay=args.weight_decay)

    with open(args.save_path/args.log_summary, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'validation_loss'])
    
    with open(args.save_path/args.log_full, "w") as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'mse_loss'])
    
    logger = TermLogger(n_epochs=args.epochs, train_size=len(train_loader), valid_size=len(val_loader))
    logger.epoch_bar.start()

    for epoch in range(args.epochs):
        logger.epoch_bar.update(epoch)

        #train for one epoch
        logger.reset_train_bar()
        train_loss = train(args, train_loader, model, optimizer, logger, training_writer)
        logger.train_writer.write(' * Avg Loss: {:.3f}'.format(train_loss))
        
        #evaluate the model in validation set
        logger.reset_valid_bar()
        errors, error_names = validate(args, val_loader, model, epoch, logger, output_writers)
        error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names, errors))
        logger.valid_writer.write(' * Avg {}'.format(error_string))

        for error, name in zip(errors, error_names):
            training_writer.add_scalar(name, error, epoch)
        
        #Choose here which is the error you want to consider
        decisive_error = errors[0]
        if best_error < 0:
            best_error = decisive_error

        #remember lowest error and save checkpoint
        is_best = decisive_error < best_error
        best_error = min(best_error, decisive_error)
        save_checkpoint(
            args.save_path, {
                'epoch': epoch + 1,
                'state_dict': model.state_dict()
            },
            is_best)
        
        with open(args.save_path/args.log_summary, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([train_loss, decisive_error])
    logger.epoch_bar.finish()

def train(args: argparse.ArgumentParser.parse_args, train_loader: DataLoader, model: nn.Module, optimizer: torch.optim.Adam, logger: TermLogger, train_writer: SummaryWriter):
    global n_iter, device
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    #switch the models to train mode
    model.train()

    end = time.time()
    logger.train_bar.update(0)

    for i, data in enumerate(train_loader):
        log_losses = i > 0 and n_iter % args.print_freq == 0   


@torch.no_grad()
def validate(args:argparse.ArgumentParser.parse_args, val_loader: DataLoader, model: nn.Module, optimizer: torch.optim.Adam, logger: TermLogger, output_writers: SummaryWriter = []):
    global device
    batch_time = AverageMeter()
    losses = AverageMeter(i=4, precision=4)
    log_outputs = len(output_writers) > 0

    #switch to evaluate mode
    model.eval()

    end = time.time()
    logger.valid_bar.update(0)

if __name__ == "__main__":
    main()