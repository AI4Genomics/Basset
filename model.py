# model.py
# --------
# censing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational purposes.# project. You are free to use and extend these projects for educational purposes.
import numpy as np
import torch
import torch.nn as nn


#note: to check dimensions, do not have to have real data (can be done with any data)
basset_net = Basset(some_arguments) # __init__ here
random_sample = np.random.randn(64, 4, 1, 600) # 64 is the batch_size
basset_net(random_sample) # forward here


class resblock_2d(nn.Module):
    """Class to return a block of 2D Resudual Networks
    Parameters:
        inputs (tensor):Input of the previous layer
        num_channels (int):Dimension parameter (in each resnet block).
        kernel_size (tuple):Size of the (2D) filters (or kernels).
    Returns:
        outputs (torch.nn.Module):'resblock_2d' module that consist of several "conv2d->BN->relu" repetitions
    """
    def __init__(self, num_channels, kernel_size=(5, 3)):
        super(resblock_2d, self).__init__()
        modules = []
        relu = nn.ReLU(True)
        for r in range(2):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(num_channels, num_channels, kernel_size, stride=(1, 1),
                              padding=(kernel_size[0]//2, kernel_size[1]//2)),  # bottleneck: (batch, width, out_chan)
                    nn.BatchNorm2d(num_channels),
                    relu
                )
            )
        self.block = nn.Sequential(*modules)
    def forward(self, inputs):
        outputs = self.block(inputs)
        return outputs
class ResNet2d(nn.Module):
    """ Resudual Network with 2D convolutions for discriminating real vs generated sequences.
    Parameters:
        inputs (tensor):Tensor of size (batch_size, vocab_size, max_seq_len) containing real or generated sequences.
        num_channels (int):Discriminator dimension parameter (in each resnet block).
        vocab_size (int):Size of the first layer input channel.
        seq_len (int):Length of the input sequence.
        num_layers (int):How many repetitions of 'resblock_2d' for discriminator.
    Returns:
        outputs (tensor):Batch of 2D tensors of values in the size (vocab_size, seq_len).
    """
    def __init__(self, num_channels=128, vocab_size=4, seq_len=600, num_classes=164, res_layers=2):
        super(ResNet2d, self).__init__()
        self.num_channels = num_channels
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.res_layers = res_layers
        self.num_classes = num_classes
        self.conv = nn.Sequential(
                nn.Conv2d(1, num_channels, (3, 1), stride=(1, 1), padding=(3//2, 0))  # (batch, width, out_chan)
            )
        resblocks = []
        for i in range(res_layers):
            resblocks.append(resblock_2d(self.num_channels))
        self.resblocks = nn.Sequential(*resblocks)
        self.prediction_layer = nn.Linear(self.vocab_size*seq_len*num_channels, num_classes)
    def forward(self, inputs):
        inputs = torch.reshape(inputs, [-1, 1, self.vocab_size, self.seq_len])
        outputs = self.conv(inputs)
        inputs = outputs
        for i in range(self.res_layers):
            outputs = 1.0*self.resblocks[i](inputs) + inputs  # where resnet idea comes into play!
            inputs = outputs
        outputs = torch.reshape(outputs, [-1, self.vocab_size*self.seq_len*self.num_channels])
        outputs = self.prediction_layer(outputs)
        return outputs



#check input dimensions
inputs = torch.reshape(inputs, [-1, 1, len(random_sample), self.seq_len])
print(inputs.shape) #See how input will look like after passed to the network and processed

#check output dimensions
outputs = self.conv(inputs)
print(outputs.shape) #Check the output shape of each layer
        """inputs = outputs
        for i in range(self.res_layers):
            outputs = 1.0*self.resblocks[i](inputs) + inputs  # where resnet idea comes into play!
            inputs = outputs
        outputs = torch.reshape(outputs, [-1, self.vocab_size*self.seq_len*self.num_channels])
        outputs = self.prediction_layer(outputs)
        return outputs"""




class Basset(nn.Module):
    """ Basset network to learn models of DNA sequence activity such as accessibility, protein binding, and chromatin state.
    Parameters:
        inputs (tensor): Batch of one-hot-encoded sequences.
	other_arguments: Should be explained later.
    Returns:
        outputs (tensor): Should be explained later.
    """
    def __init__(self, other_arguments=None):
        super(Basset, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 100, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.block2 = nn.Sequential(
            nn.Conv2d(100, 84, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, inputs):
        out = self.block1(inputs)
        out = self.block2(out)

        # Flatten the output of block2
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
