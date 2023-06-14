import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import argparse
import time
import csv
import numpy as np
from datasets import augmentations
from utils import save_checkpoint, create_saving_dir, none_or_str
from tensorboardX import SummaryWriter
from path import Path
from logger import TermLogger, AverageMeter
from models.force_estimator import ForceEstimator
from datasets.vision_state_dataset import VisionStateDataset
import os

parser = argparse.ArgumentParser(description='Vision and robot state based force estimator using different architectures',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset', default='chua', choices=['img2force', 'chua', 'mixed'], help='available training dataset')
parser.add_argument('--type', default='vs', choices=['v', 'vs'], type=str, help='model type it can be vision only (v) or vision and state (vs)')
parser.add_argument('--architecture', choices=['cnn', 'vit', 'fc'], type=str, help='')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M', help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--seed', default=0, type=int, help='seed for random function, and network initialization')
parser.add_argument('--log-summary', default='log_summary.csv', metavar='PATH', help='csv to save the per-epoch train and valid stats')
parser.add_argument('--log-full', default='log_full.csv', metavar='PATH', help='csv to save the progress per gradient during training')
parser.add_argument('--log-output', action='store_true', help='will log force estimation outputs at validation')
parser.add_argument('--pretrained', default=None, metavar='PATH', help='path to pretrained model')
parser.add_argument('--name', dest='name', type=str, required=True, help='name of the experiment, checkpoint are stored in checkpoint/name')
parser.add_argument('-r', '--rmse-loss-weight', default=5.0, type=float, help='weight for rroot mean square error loss')
parser.add_argument('-g', '--gd-loss-weight', default=0.5, type=float, help='weight for gradient difference loss')
parser.add_argument('--train-type', choices=['random', 'geometry', 'color', 'structure', 'stiffness', "position"], default='random', type=str, help='training type for comparison')
parser.add_argument('--occlude-param', choices=["force_sensor", "robot_p", "robot_o", "robot_v", "robot_w", "robot_q", "robot_vq", "robot_tq", "robot_qd", "robot_tqd", "None"], help="choose the parameters to occlude")
parser.add_argument('--include-depth', action='store_true')
parser.add_argument('--att-type', default=None, help="adding attention layers to the network")
parser.add_argument('--recurrency', action='store_true')

best_error = -1
n_iter = 0
num_samples = 1000

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.autograd.set_detect_anomaly(True) 

def main():
    global best_error, n_iter, device
    args = parser.parse_args()
    save_path = Path(args.name)
    root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    checkpoint_dir = root/'checkpoints'
    args.save_path = create_saving_dir(checkpoint_dir, save_path, args.architecture, args.include_depth, args.dataset, args.recurrency, args.att_type)
    # args.save_path = '/nfs/home/mreyzabal/checkpoints/{}/{}'.format('chua' if args.chua else 'img2force', )/save_path/timestamp
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
    
    bright = augmentations.BrightnessContrast(contrast=2., 
                                              brightness=12.)
    
    # noise = augmentations.GaussianNoise(noise_factor = 0.25)

    train_transform = augmentations.Compose([
        # augmentations.RandomHorizontalFlip(),
        # augmentations.RandomVerticalFlip(),
        # augmentations.RandomRotation(),
        augmentations.CentreCrop(),
        augmentations.SquareResize(),
        augmentations.ArrayToTensor(),
        normalize,
        # noise
    ]) if args.dataset=="chua" else augmentations.Compose([
        # augmentations.RandomHorizontalFlip(),
        # augmentations.RandomVerticalFlip(),
        # augmentations.RandomRotation(),
        augmentations.CentreCrop(),
        augmentations.SquareResize(),
        bright,
        augmentations.ArrayToTensor(),
        normalize,
    ])

    val_transform = augmentations.Compose([
        augmentations.CentreCrop(),
        augmentations.SquareResize(),
        augmentations.ArrayToTensor(),
        normalize
    ]) if args.dataset=="chua" else augmentations.Compose([
        augmentations.CentreCrop(),
        augmentations.SquareResize(),
        bright,
        augmentations.ArrayToTensor(),
        normalize,
    ])
    
    if args.recurrency:
        recurrency_size = 5
    else:
        recurrency_size = 1

    occ_param = none_or_str(args.occlude_param)
    print("=> Getting scenes from the dataset '{}'".format(args.dataset))
    print("=> Choosing the correct dataset for choice {}...".format(args.train_type))
    
    train_dataset = VisionStateDataset(mode="train", transform=train_transform, seed=args.seed, train_type=args.train_type, recurrency_size=recurrency_size, load_depths=args.include_depth, occlude_param=occ_param, dataset=args.dataset)
    val_dataset = VisionStateDataset(mode="val", transform=val_transform, seed=args.seed, train_type=args.train_type, recurrency_size=recurrency_size, load_depths=args.include_depth, occlude_param=occ_param, dataset=args.dataset)

    print('{} samples found in {} train scenes'.format(len(train_dataset), len(train_dataset.folder_index) if args.dataset=="chua" else len(train_dataset.scenes)))
    print('{} samples found in {} validation scenes'.format(len(val_dataset), len(val_dataset.folder_index) if args.dataset=="chua" else len(val_dataset.scenes)))

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # Create the model
    if args.type == "v":
        state_size = 0
    else:
        state_size = 54

    print("=> Creating the {} model with {} encoder and {}...".format("vision & state" if args.type=="vs" else "vision", args.architecture, "recurrency" if args.recurrency else "linear"))

    model = ForceEstimator(architecture=args.architecture,
                           recurrency=args.recurrency,
                           pretrained=False,
                           include_depth=args.include_depth,
                           att_type=args.att_type,
                           state_size=state_size)

    model.to(device)

    #Load parameters
    if args.pretrained:
        print("=> Using pre-trained weights for ViT")
        weights_vit = torch.load(args.pretrained)
        model.load_state_dict(weights_vit['state_dict'], strict=False)
    
    print("=> Setting Adam optimizer")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3 if args.type=="vs" else 1e-4, betas=(args.momentum, args.beta), weight_decay=args.weight_decay)
    
    #Initialize losses
    mse = nn.MSELoss()

    with open(args.save_path/args.log_summary, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss_vit', 'validation_loss_vit'])
    
    with open(args.save_path/args.log_full, "w") as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss_vit', 'mse_loss_vit'])
    
    logger = TermLogger(n_epochs=args.epochs, train_size=len(train_loader), valid_size=len(val_loader))
    logger.epoch_bar.start()

    for epoch in range(args.epochs):
        logger.epoch_bar.update(epoch)

        #train for one epoch
        logger.reset_train_bar()
        train_loss = train(args, train_loader, model, optimizer, logger, training_writer, mse)
        logger.train_writer.write(' * Avg Loss: {:.3f}'.format(train_loss))
        
        #evaluate the model in validation set
        logger.reset_valid_bar()
        errors, error_names = validate(args, val_loader, model, logger, output_writers)
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
                'state_dict': model.state_dict(),
            },
            is_best)
        
        with open(args.save_path/args.log_summary, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([train_loss, errors[0]])
    logger.epoch_bar.finish()


def train(args: argparse.ArgumentParser.parse_args, train_loader: DataLoader, model: nn.Module, optimizer: torch.optim.Adam, logger: TermLogger, train_writer: SummaryWriter, mse: nn.MSELoss):
    global n_iter, device
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(i=2,precision=4)
    w1, w2 = args.rmse_loss_weight, args.gd_loss_weight
    l1_lambda = 1e-3
    eps = 1e-5

    #switch the vit_models to train mode
    model.train()

    end = time.time()
    logger.train_bar.update(0)

    for i, data in enumerate(train_loader):
        if i > num_samples:
            break
        log_losses = i > 0 and n_iter % args.print_freq == 0
        data_time.update(time.time() - end)
        img = [im.to(device) for im in data['img']] if args.recurrency else data['img'].to(device)
        forces = data['forces'].to(device).float()
        state = data['robot_state'].squeeze(1).to(device).float() if args.type=="vs" else None           
        
        pred_forces = model(state) if args.architecture=="fc" else model(img, state)
        mse_loss = (pred_forces - forces).pow(2).mean() + eps
        # Add L1 regularization
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        loss = w1 * mse_loss + l1_lambda * l1_norm

        if log_losses:
            train_writer.add_scalar('MSE_ViT', mse_loss.item(), n_iter)
            train_writer.add_scalar('Loss_ViT', loss.item(), n_iter)
        
        # record loss and EPE
        losses.update([loss.item(), mse_loss.item()], args.batch_size)

        # compute gradient and do Adam step for vit
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        with open(args.save_path/args.log_full, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([loss.item(), mse_loss.item()])

        logger.train_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.train_writer.write('Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))
        
        n_iter += 1
    
    return losses.avg[1]

@torch.no_grad()
def validate(args:argparse.ArgumentParser.parse_args, val_loader: DataLoader, model: nn.Module, logger: TermLogger, output_writers: SummaryWriter = []):
    global device
    batch_time = AverageMeter()
    losses = AverageMeter(i=1, precision=4)
    log_outputs = len(output_writers) > 0

    #switch to evaluate mode
    model.eval()

    end = time.time()
    logger.valid_bar.update(0)

    for i, data in enumerate(val_loader):
        img = [im.to(device) for im in data['img']] if args.recurrency else data['img'].to(device)
        forces = data['forces'].to(device).float()
        state = data['robot_state'].squeeze(1).to(device).float() if args.type == "vs" else None            
        
        pred_forces = model(state) if args.architecture=="fc" else model(img, state)
        loss = torch.sqrt(((forces - pred_forces) ** 2).mean())
        losses.update([loss.item()])
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.valid_writer.write('Valid: Time {} Loss {}'.format(batch_time, losses))
    
    logger.valid_bar.update(len(val_loader))
    return losses.avg, ['Loss']

    
if __name__ == "__main__":
    main()