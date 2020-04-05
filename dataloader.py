# The Dataloader is responsible to take the result of the Dataset class and prepare it for training by batching the inputs
#   or apply shuffling if we pass the train data or test data to it (ie. if train, shuffle argument should be True).
# It can also apply some sort of transformation to the data if we ask to
#   (eg, rotating the input images with slight angles in each training iteration so the network would see them as new images and won't overfit)

#train_dataloader = DataLoader(train_dataset, batchsize, shuffle=False, num_workers=data_workers)


from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader

def create_dataset(seq, labels, n):
    """
    Slice the first n seq/labels and create a torch.utils.data.DataLoader.

    Args:
    :param seq: numpy array of sequences
    :param labels: list of labels associated with the sequences
    :param n: the number of seq/labels to slice

    :return: A torch.utils.data.TensorDataset to be used with a torch.utils.data.DataLoader.
    """
    sequences = torch.tensor(seq[:n], dtype=torch.float)
    labels = torch.tensor(labels[:n], dtype=torch.long)
    dataset = TensorDataset(seq, labels)
    return dataset

n = 100
batch_size = 32
train_dataset = create_dataset(train_in, train_out, n)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=TRUE)
#By setting drop_last=False, the last incomplete batch is kept if the dataset size is not divisible by batch_size.

epochs = 2
for epoch in range(epochs):
    print("Epoch {}/{}:".format(epoch+1, epochs))
    for i, (x,y) in enumerate(train_dataloader):
        print("     batch {}/{} or {} examples.".format(i+1, int(np.ceil(n/batch_size)), y.size(0)))
#At every iteration, the dataloader returns a mini-batch of batch_size input-label pairs (x, y).