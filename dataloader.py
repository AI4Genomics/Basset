# The Dataloader is responsible to take the result of the Dataset class and prepare it for training by batching the inputs
#   or apply shuffling if we pass the train data or test data to it (ie. if train, shuffle argument should be True).
# It can also apply some sort of transformation to the data if we ask to
#   (eg, rotating the input images with slight angles in each training iteration so the network would see them as new images and won't overfit)

# Goal: extract 3 separate files -- training, testing, and validation sets
#train_dataloader = DataLoader(train_dataset, batchsize, shuffle=False, num_workers=data_workers)

<<<<<<< HEAD
from torchim.utils.data import DataLoader

#TODO: SIMPLY EXTRACT THE TRAINING, TESTING AND VALIDATION SET ALREADY GENERATED FROM THE PREPROCESSING USING COMMAND LINE

import numpy as np
import h5py

import os
import h5py
import preprocess
import dataset
from torch.utils.data import DataLoader

with h5py.File('er.h5', 'r') as hdf:
    ls = list(hdf.keys())
    print('List of datasets in this file: \n', ls)

    # getting the shape of the keys in the HDF5 file
    train_dataset = hdf.get('train_in') # returns the value of train_in key
    train_dataset_out = hdf.get('train_out')
    test_dataset = hdf.get('test_in')
    test_dataset_out = hdf.get('test_out')
    validation_dataset = hdf.get('valid_in')

    print("Shape of the input training data is:", train_dataset.shape)
    print("Shape of the out training data is:", train_dataset_out.shape)
    print("Shape of the input testing data is:", test_dataset.shape)
    print("Shape of the output testing data is:", test_dataset_out.shape)
    print("Shape of the validation data is:", validation_dataset.shape)
    print(hdf.get('test_headers')[150])
    print("\n")

    # listing all of the keys in our file
    print("They keys in our HDF5 file are: \n")
    for key in hdf.keys():
        print(key)
        # all of the keys in the HDF5 file

    # creating new HDF5 datasets for our train, test, and validation data
    print("\n")
    print("Our train/test/validation objects: \n")
    train_in = hdf["train_in"]
    print("Dataset train_in: ", train_in)
    test_in = hdf["test_in"]
    print("Dataset test_in:", test_in)
    valid_in = hdf["valid_in"]
    print("Dataset valid_in:", valid_in)

#TODO NEXT STEPS FOR DATALOADER
train_dataset_sizes = len(train_dataset)
num_train_samples = int(0.8 * train_dataset_sizes)
num_valid_samples = train_dataset_sizes - num_train_samples
num_test_samples = len(test_dataset)

print('# of train examples: {}'.format(num_train_samples))
print('# of valid examples: {}'.format(num_valid_samples))
print('# of test examples: {}'.format(num_test_samples))

batch_size = 32
epochs = 2

train_loader = DataLoader(dataset=train_dataset,
                          sampler=BassetDataset(num_train_samples, 0),
                          batch_size=batch_size,
                          drop_last=False, #By setting drop_last=False, the last incomplete batch is kept if the dataset size is not divisible by batch_size.
                          shuffle=False)
for epoch in range(epochs):
    print("Epoch {}/{}:".format(epoch+1, epochs))
    for i, (x,y) in enumerate(train_loader):
        print("     batch {}/{} or {} examples.".format(i+1, int(np.ceil(n/batch_size)), y.size(0)))
#At every iteration, the dataloader returns a mini-batch of batch_size input-label pairs (x, y).

valid_loader = DataLoader(dataset=train_dataset,
                          sampler=BassetDataset(
                              num_valid_samples, num_train_samples),
                          batch_size=batch_size,
                          shuffle=False)
for epoch in range(epochs):
    print("Epoch {}/{}:".format(epoch+1, epochs))
    for i, (x,y) in enumerate(valid_loader):
        print("     batch {}/{} or {} examples.".format(i+1, int(np.ceil(n/batch_size)), y.size(0)))

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False)
for epoch in range(epochs):
    print("Epoch {}/{}:".format(epoch+1, epochs))
    for i, (x,y) in enumerate(test_loader):
        print("     batch {}/{} or {} examples.".format(i+1, int(np.ceil(n/batch_size)), y.size(0)))
