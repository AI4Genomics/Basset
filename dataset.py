# Dataset class would split the original data into train, test, validation splits.
# The original data is the result of the preprocessing step which can be stored in a text, csv, hdf5, etc file

import torch
import numpy as np
from torch.utils.data import Dataset

seed = 1234
np.random.seed(seed) # Set the random seed of numpy for the data split.

use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")

print("Torch version: ", torch.__version__)
print("GPU Available: {}".format(use_gpu))



train_dataset = BassetDataset(data_path, file_name, 'train_in', 'train_out')


class BassetDataset(Dataset):
    # Initializes the BassetDataset
    def __init__(self, path, f5name, sequences, labels, transform=None):
        # Create a list called <samples> which will store all the sequences/datapoints from HDF5 file
        self.samples = h5py.File(os.path.join(path, f5name))
        #
        self.train = self.samples['.'][sequences]
        self.test = self.samples['.'][labels]
        self.samples_len, self.n_nucleotides, _, self.seq_len = self.test.shape
        self.output_len, self.n_output = self.test.shape
        assert self.samples_len == self.output_len  # testing that samples_len & output_len are same == self.test_shape & self.y.shape are same
        self.train = self.train[:].reshape([self.samples_len, self.n_nucleotides, self.seq_len])

        print(samples
        "Input shape: {self.train.shape}")
        print(self.train[1])
        # samples.close()

    # Returns the size of the dataset
    def __len__(self):
        return self.samples_len

    # Returns a sample from the dataset given an index
    def __getitem__(self, index):
        """
        This method gets its indexed item from the dataset
        """
        return self.train[index], self.test[index]

    def cleanup(self):
        self.samples.close()

