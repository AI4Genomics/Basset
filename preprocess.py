import numpy as np
import h5py

with h5py.File('er.h5', 'r') as hdf:
    ls = list(hdf.keys())
    print('List of datasets in this file: \n', ls)
    sequences = hdf.get('train_in')
    labels = hdf.get('train_out')
    print("Shape of the input training data is:", train_in.shape)
    print("Shape of the out training data is:", train_out.shape)
    print(hdf.get('test_headers')[150])

#create a validation dataset from original training dataset
def partition_dataset(seqs, labels, valid_ratio=0.1, shuffle=True, seed=1234):
    """
    Args:
       imgs: numpy array representing the sequence set from which the partitioning is made.
       labels: the labels associated with the provided sequences.
       valid_ratio (optional): the portion of the data that will be used in
          the validation set. Default: 0.1.
       shuffle (optional): whether or not to shuffle the data. Default: True.
       seed (optional): the seed of the numpy random generator: Default: 1234.

    Return:
       A tuple of 4 elements (train_seqs, train_labels, valid_seqs, valid_labels)
       where:
          train_seqs: a numpy array of sequences for the training set.
          train_labels: labels associated with the sequences in the training set.
          valid_seqs: a numpy array of sequences for the validation set.
          valid_labels: labels associated with the sequences in the validation set.

    """
    if shuffle:
        np.random.seed(seed)  # Set the random seed of numpy.
        indices = np.random.permutation(seqs.shape[0])
        #<np.random.permutation> Randomly permute a sequence. If x is a multi-dimensional array, it is only shuffled along its first index.
         #<.shape> returns dimensions of the array
           #if Y has n rows and m cols, Y.shape is (n,m)
         #<seqs.shape[0]> is n of seqs
    else:
        indices = np.arange(seqs.shape[0]) #<np.arange> Return evenly spaced values within a given interval

    train_idx, valid_idx = np.split( #split indices into train and valid index based on valid_ratio
        indices,
        [int((1.0 - valid_ratio) * len(indices))]
    )
    train_seqs, valid_seqs = seqs[train_idx], seqs[valid_idx]
    tgt = np.array(labels)
    train_labels, valid_labels = tgt[train_idx].tolist(), tgt[valid_idx].tolist()
    return train_seqs, train_labels, valid_seqs, valid_labels

train_seq, train_labels, valid_seq, valid_labels = partition_dataset(sequences, labels)