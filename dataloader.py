# The Dataloader is responsible to take the result of the Dataset class and prepare it for training by batching the inputs
#   or apply shuffling if we pass the train data or test data to it (ie. if train, shuffle argument should be True).
# It can also apply some sort of transformation to the data if we ask to
#   (eg, rotating the input images with slight angles in each training iteration so the network would see them as new images and won't overfit)

#train_dataloader = DataLoader(train_dataset, batchsize, shuffle=False, num_workers=data_workers)

import preprocess
import dataset
from torch.utils.data import DataLoader

#TODO: SIMPLY EXTRACT THE TRAINING, TESTING AND VALIDATION SET ALREADY GENERATED FROM THE PREPROCESSING USING COMMAND LINE
with h5py.File('er.h5', 'r') as hdf:
    ls = list(hdf.keys())
    print('List of datasets in this file: \n', ls)
    train_dataset = hdf.get('train_in')
    test_dataset = hdf.get('train_out')
    valid_dataset =
    print("Shape of the input training data is:", train_in.shape)
    print("Shape of the out training data is:", train_out.shape)
    print(hdf.get('test_headers')[150])

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
