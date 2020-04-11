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
from pytz import timezone
from torch.utils.data import DataLoader
from dataset import BassetDataset
#from model import BassetNet

pp = pprint.PrettyPrinter(indent=4)

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='./data', help="Path to the dataset directory (default: './data'.")
parser.add_argument('--file_name', type=str, default='sample_dataset.h5', help='Name of the h5 file already preprocessed in the preprocessing step (default: sample_dataset.h5).')
parser.add_argument('--batch_size', type=int, default=256, help='Defines the batch size for training phase (default: 64).')
parser.add_argument('--nb_epochs', type=int, default=2, help='Defines the maximum number of epochs the network needs to train (default: 200).')
parser.add_argument('--omptimizer', type=str, default='adam', help='The method used for optimization of the model (default: "adam").') 
parser.add_argument('--validate', type=bool, default=True, help='Whether to use validation set')
parser.add_argument('--learning_rate', type=float, default=0.004, help='Learning rate for the optimizer (default: 0.004)')
parser.add_argument('--beta1', type=float, default=0.5, help='"beta1" for the optimizer')
parser.add_argument('--seed', type=int, default=313, help='Seed used for reproducibility')
args = parser.parse_args()

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Arguments are:\n")
pp.pprint(vars(args)) # better to use '--help' instead (try now!) to get the list of args; please remove after understanding
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

basset_dataset_train = BassetDataset(path='./data/', f5name='sample_dataset.h5', split='train')
print("The number of samples in {} split is {}.\n".format('train', len(basset_dataset_train)))

basset_dataset_valid = BassetDataset(path='./data/', f5name='sample_dataset.h5', split='valid')
print("The number of samples in {} split is {}.\n".format('valid', len(basset_dataset_valid)))

basset_dataset_test = BassetDataset(path='./data/', f5name='sample_dataset.h5', split='test')  # we do not need these here, we need then in the 'test.py' (just to let you know)
print("The number of samples in {} split is {}.".format('test', len(basset_dataset_test)))
print("The first 10 ids of test samples are:\n  {}\n".format("\n  ".join(basset_dataset_test.ids[:10])))

basset_dataloader_train = DataLoader(basset_dataset_train, batch_size=args.batch_size, drop_last=True, shuffle=True, num_workers=1)
basset_dataloader_valid = DataLoader(basset_dataset_valid, batch_size=len(basset_dataset_valid), drop_last=False, shuffle=False, num_workers=1)

for n_epoch in range(args.nb_epochs):
    # training
    print("TRAINING (EPOCH {}):".format(n_epoch+1))
    print("=======================\n")
    for n_batch, batch_samples in enumerate(basset_dataloader_train):
        seqs, trgs = batch_samples[0], batch_samples[1]
        print("Shape of the batch for input (batch_{}/epoch_{}): {}".format(n_batch, n_epoch+1, seqs.shape))
        print("Shape of the batch for output (batch_{}/epoch_{}): {}\n".format(n_batch, n_epoch+1, trgs.shape))
        ### IMPORTANT: here, seqs whold be fed to the BassetNet (imported above) and output of it would be compare against trgs for network optimization
    
    # validation
    print("VALIDATION:")
    seqs, trgs = next(iter(basset_dataloader_valid))
    print("Shape of the for input of validation data (epoch_{}): {}".format(n_epoch+1, seqs.shape))
    print("Shape of the batch for output of validation data (epoch_{}): {}\n".format(n_epoch+1, trgs.shape))
    ### IMPORTANT: here, we look at the results to see what hype-parameters are working well, to keep them    
