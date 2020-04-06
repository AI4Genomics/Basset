# Dataset class would split the original data into train, test, validation splits.
# The original data is the result of the preprocessing step which can be stored in a text, csv, hdf5, etc file

from torch.utils.data import Dataset

train_dataset = BassetDataset(data_path, file_name, 'train_in', 'train_out')

class BassetDataset(Dataset):
    def __init__(self, path, f5name, X_dataset_name, y_dataset_name, transform=None):

        self.f = h5py.File(os.path.join(path, f5name) )
        self.X = self.f['.'][X_dataset_name]
        self.y = self.f['.'][y_dataset_name]
        self.samples_len, self.n_nucleatides, _, self.seq_len = self.X.shape
        self.output_len, self.n_output = self.y.shape
        assert self.samples_len == self.output_len
        self.X = self.X[:].reshape([self.samples_len, self.n_nucleatides, self.seq_len])
        
	print(f"Input shape: {self.X.shape}")
        print(self.X[1])
        #f.close()

    def __len__(self):
        return self.samples_len

    def __getitem__(self, index):
        """
        This method gets it index'th item from the dataset
        """
        return self.X[index], self.y[index]

    def cleanup(self):
        self.f.close()
