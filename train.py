# train.py
# --------
# censing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational purposes.# project. You are free to use and extend these projects for educational purposes.

import os
from os import path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import datetime
import random
import re
import pprint
import requests
import logomaker # https://github.com/jbkinney/logomaker/tree/master/logomaker/tutorials (should be moved to the util.py)
import argparse
from util import cal_iter_time
from pytz import timezone
from torch.utils.data import DataLoader
from dataset import BassetDataset
from model import Basset

tz = timezone('US/Eastern')
pp = pprint.PrettyPrinter(indent=4)

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='./data', help="Path to the dataset directory (default: './data')")
parser.add_argument('--file_name', type=str, default='sample_dataset.h5', help='Name of the h5 file already preprocessed in the preprocessing step (default: sample_dataset.h5)')
parser.add_argument('--log_dir', default='log/', help='Base log folder (create if it does not exist')
parser.add_argument('--log_name', default='basset_train', help='name to use when logging this model')
parser.add_argument('--batch_size', type=int, default=64, help='Defines the batch size for training phase (default: 64)')
parser.add_argument('--nb_epochs', type=int, default=200, help='Defines the maximum number of epochs the network needs to train (default: 200)')
parser.add_argument('--optimizer', type=str, default='adam', help="The algorithm used for the optimization of the model (default: 'adam')")
parser.add_argument('--validate', type=bool, default=True, help='Whether to use validation set')
parser.add_argument('--learning_rate', type=float, default=0.004, help='Learning rate for the optimizer (default: 0.004)')
parser.add_argument('--beta1', type=float, default=0.5, help="'beta1' for the optimizer")
parser.add_argument('--seed', type=int, default=313, help='Seed for reproducibility')
args = parser.parse_args()

# save args in the log folder

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Arguments are:")
pp.pprint(vars(args))
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

# sets device for model and PyTorch tensors
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set RNG
seed = args.seed
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if device.type=='cuda':
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# create 'log_dir' folder (if it does not exist already)
os.makedirs(args.log_dir, exist_ok=True)

basset_dataset_train = BassetDataset(path='./data/', f5name='sample_dataset.h5', split='train')
print("The number of samples in {} split is {}.\n".format('train', len(basset_dataset_train)))

basset_dataset_valid = BassetDataset(path='./data/', f5name='sample_dataset.h5', split='valid')
print("The number of samples in {} split is {}.\n".format('valid', len(basset_dataset_valid)))

# these three lines should be move to 'test.py', they do not belong to the training process!
basset_dataset_test = BassetDataset(path='./data/', f5name='sample_dataset.h5', split='test')  # we do not need these here, we need then in the 'test.py' (just to let you know)
print("The number of samples in {} split is {}.".format('test', len(basset_dataset_test)))
print("The first 10 ids of test samples are:\n  {}\n".format("\n  ".join(basset_dataset_test.ids[:10])))

# using default pytorch DataLoaders
basset_dataloader_train = DataLoader(basset_dataset_train, batch_size=args.batch_size, drop_last=True, shuffle=True, num_workers=1)
basset_dataloader_valid = DataLoader(basset_dataset_valid, batch_size=len(basset_dataset_valid), drop_last=False, shuffle=False, num_workers=1)

# basset network instantiation
basset_net = Basset()

# cost function
criterion = nn.BCEWithLogitsLoss()

# setup optimizer & scheduler
"""if args.optimizer=='adam':
    optimizer = optim.Adam(list(basset_net.parameters()), lr=args.learning_rate, betas=(args.beta1, 0.999))
elif args.optimizaer=='rmsprop':
    optimizaer = optim.RMSprop(list(basset_net.parameters()), lr=args.learning_rate)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)  # use an exponentially decaying learning rate """ 

# keeping track of the time
start_time = datetime.datetime.now(tz)
former_iteration_endpoint = start_time
print("~~~~~~~~~~~~~ TIME ~~~~~~~~~~~~~~")
print("Time started: {}".format(start_time.strftime("%Y-%m-%d %H:%M:%S")))
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

# main training loop
for n_epoch in range(args.nb_epochs):
    # training
    print("TRAINING (EPOCH {}):".format(n_epoch+1))
    print("=======================")
    for n_batch, batch_samples in enumerate(basset_dataloader_train):
        seqs, trgs = batch_samples[0], batch_samples[1]
        print("Shape of the batch for training input (batch_{}/epoch_{}): {}".format(n_batch+1, n_epoch+1, seqs.shape))
        print("Shape of the batch for training output (batch_{}/epoch_{}): {}\n".format(n_batch+1, n_epoch+1, trgs.shape))
        ### IMPORTANT: here, seqs whold be fed to the BassetNet (imported above) and output of it would be compare against trgs for network optimization
        # predictions = basset_net(seqs)
        # err = compare(predictions vs. trgs)
        # err.backwards
        # basset_network.step()
    
    # validation
    if args.validate:
        print("VALIDATION:")
        print("------------------")
        seqs, trgs = next(iter(basset_dataloader_valid))
        print("Shape of the input for the validation data (epoch_{}): {}".format(n_epoch+1, seqs.shape))
        print("Shape of the output for the validation data (epoch_{}): {}\n".format(n_epoch+1, trgs.shape))
        ### IMPORTANT: here, we look at the results to see what hype-parameters are working well, to keep them
        # report = compare(predictions vs. trgs)
    
    # show/save stats of the results in the log folder
    # checkpoint the basset_net in the log folder
    former_iteration_endpoint = cal_iter_time(former_iteration_endpoint, tz)

