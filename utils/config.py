import os
import torch
import random
import numpy as np
import setproctitle
from argparse import ArgumentParser
from torch import cuda, device as Device

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
setproctitle.setproctitle("txk-NoduleSeg")

parser = ArgumentParser(description="Image Segmentation of Lung Nodules")
parser.add_argument("--seg_model", type=str, default="SwinUNETR",
                    help="name of the segmentation model")
parser.add_argument("--device", type=Device, help="runtime device",
                    default=Device("cuda" if cuda.is_available() else "cpu"))

############################ save path ############################
parser.add_argument("--data_root", type=str, default="../DATASETS/LUNA-Seg",
                    help="root of the dataset")
parser.add_argument("--json_name", type=str, default="dataset.json",
                    help="name of json file in cross validation")
parser.add_argument("--num_folds", type=int, default=5,
                    help="number of folds in cross validation")
parser.add_argument("--fold", type=int, default=0,
                    help="fold of cross validation")

parser.add_argument("--model_save_dir", type=str, default="../models/LUNA-16",
                    help="save path of trained models")
parser.add_argument("--trained_model", type=str, default="best_model_f1.pth",
                    help="filename of pretrained model")
parser.add_argument("--log_save_dir", type=str, default="logs",
                    help="save path of runtime logs")
parser.add_argument("--seg_save_dir", type=str, default="segmentation",
                    help="filename of segmentation results")

############################ training ############################
parser.add_argument("--batch_size", type=int, default=4,
                    help="batch size of training")
parser.add_argument("--max_epochs", type=int, default=100,
                    help="max number of training epochs")
parser.add_argument("--val_freq", type=int, default=1,
                    help="validation frequency")
parser.add_argument("--num_workers", type=int, default=8,
                    help="number of workers")

parser.add_argument("--optim_name", type=str, default="AdamW",
                    help="name of optimizer")
parser.add_argument("--optim_lr", type=float, default=3e-4,
                    help="learning rate of optimizer")
parser.add_argument("--reg_weight", type=float, default=1e-5,
                    help="regularization weight")
parser.add_argument("--lrschedule", type=str, default="WarmUpCosine",
                    help="name of learning rate scheduler")
parser.add_argument("--warmup_steps", type=int, default=10,
                    help="number of learning rate warmup steps")

######################### preprocessing #########################
parser.add_argument("--roi_x", type=int, default=64,
                    help="roi size in X direction")
parser.add_argument("--roi_y", type=int, default=64,
                    help="roi size in Y direction")
parser.add_argument("--roi_z", type=int, default=64,
                    help="roi size in Z direction")

parser.add_argument("--in_channels", type=int, default=1,
                    help="number of input channels")
parser.add_argument("--out_channels", type=int, default=2,
                    help="number of output channels")

SEED = 35202
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

args = parser.parse_args()