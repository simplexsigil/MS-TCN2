#!/usr/bin/python2.7

import torch
from model import Trainer
from batch_gen import BatchGenerator
import os
import argparse
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--dataset', default="breakfast")
parser.add_argument('--split', default='1')

parser.add_argument('--features_dim', default='2048', type=int)
parser.add_argument('--bz', default='1', type=int)
parser.add_argument('--lr', default='0.0005', type=float)
parser.add_argument('--target_fps', default='10', type=int)


parser.add_argument('--num_f_maps', default='64', type=int)

# Need input
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--num_layers_PG', type=int, default=11,)
parser.add_argument('--num_layers_R', type=int, default=10,)
parser.add_argument('--num_R', type=int, default=3,)

args = parser.parse_args()

label2idx = {
    "walk": 0,
    "fall": 1,
    "fallen": 2,
    "sit_down": 3,
    "sitting": 4,
    "lie_down": 5,
    "lying": 6,
    "stand_up": 7,
    "standing": 8,
    "other": 9,
}

idx2label = {v: k for k, v in label2idx.items()}

num_epochs = args.num_epochs
features_dim = args.features_dim
bz = args.bz
lr = args.lr

num_layers_PG = args.num_layers_PG
num_layers_R = args.num_layers_R
num_R = args.num_R
num_f_maps = args.num_f_maps

dataset_root = "/pfs/work8/workspace/ffhk/scratch/kf3609-ws/data/omnifall"
vid_list_file = os.path.join(dataset_root, "segmentation_annotations", "splits", "train_cs.csv")
vid_list_file_tst = os.path.join(dataset_root, "segmentation_annotations", "splits", "test_cs.csv")
features_path = dataset_root
gt_path = os.path.join(dataset_root, "segmentation_annotations", "labels", "full.csv")


model_dir = "./models/"+args.dataset+"/split_"+args.split
results_dir = "./results/"+args.dataset+"/split_"+args.split

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

num_classes = len(label2idx)
trainer = Trainer(num_layers_PG, num_layers_R, num_R, num_f_maps, features_dim, num_classes, args.dataset, args.split)
if args.action == "train":
    batch_gen = BatchGenerator(num_classes, label2idx, gt_path, features_path, args.target_fps)
    batch_gen.read_data(vid_list_file)
    trainer.train(model_dir, batch_gen, num_epochs=num_epochs, batch_size=bz, learning_rate=lr, device=device)

if args.action == "predict":
    trainer.predict(model_dir, results_dir, features_path, vid_list_file_tst, num_epochs, label2idx, device, args.target_fps)

