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

class BassetDataset(Dataset):
    # Initializes the BassetDataset
    def __init__(self, path, f5name, split='train', transform=None):
        """
        Args:
            :param path: path to HDF5 file
            :param f5name: HDF5 file name
            :param sequences: input dataset name
            :param labels: output dataset name
            :param transform (callable, optional): Optional transform to be applied on a sample
        """

        # Create a list called <samples> which will store all the sequences/datapoints from HDF5 file
        self.samples = h5py.File(os.path.join(path, f5name))
        #
        self.train = self.samples['.'][sequences]
        self.test = self.samples['.'][labels]
        self.samples_len, self.n_nucleotides, _, self.seq_len = self.test.shape
        self.output_len, self.n_output = self.test.shape
        assert self.samples_len == self.output_len  # testing that samples_len & output_len are same == self.test_shape & self.y.shape are same
        self.train = self.train[:].reshape([self.samples_len, self.n_nucleotides, self.seq_len])

        print("samples input shape: {}".format(self.train.shape))
        print(self.train[1])
        # samples.close()

    # Returns the size of the dataset
    def __len__(self):
        return self.samples_len

    # Returns a sample from the dataset given an index
    def __getitem__(self, index):
        """
        This method gets its idx-th item from the dataset
        """
        return self.train[index], self.test[index]

    def cleanup(self):
        self.samples.close()


if __name__ == '__main__': # Notice: this helps to run this script independent from the rest of the project as well
    import argparse
    parser = argparse.ArgumentParser() # Notice: list of the arguments for this script, feel free to add more if you think it is needed
    parser.add_argument('--path', type=str, required=True, help='Path to the dataset directory.')
    parser.add_argument('--file_name', type=str, required=True, help='Name of the h5 file already preprocessed.')
    parser.add_argument('--split', type=str, default='train', help='Defines what data split to work with (default=tarin).')
    args = parser.parse_args()
    
    print("Arguments are: ".format(args)) # Notice: this line needs to be removed later; use 'python dataset.py --help' instead (try now!) to get the list of args
    
    if args.split=='train': # Notive, you need to complete this based on your preference 
        print('Preparing Basset Dataset for training phase...')
    else:
        pass
    print('Done')    
    
    basset_dataset = BassetDataset(args.path, args.file_name, args.split)#, 'train_in', 'train_out') # Notice: we need something similar to this in train.py & test.py (after importing BassetDataset)
    # Notice: you must check some tensor shapes and other stuff here (for basset_dataset[0] & basset_dataset[1]) to make sure it's done right above
