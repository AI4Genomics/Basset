# model.py
# --------
# censing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational purposes.# project. You are free to use and extend these projects for educational purposes.

import numpy as np
import torch
import torch.nn as nn

#TODO: LOOK OVER SOLUTIONS <CLASS RESBLOCK_2D(NN.MODULE):> SECTION
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
        pass

    def forward(self, inputs):
        pass

   
class resblock_1d(nn.Module):
    """Class to return a block of 1D Resudual Networks
    
    Parameters:
        inputs (tensor):Input of the previous layer
        num_channels (int):Dimension parameter (in each resnet block).
        kernel_size (int):Size of the (1D) filters (or kernels).

    Returns:
        outputs (torch.nn.Module):'resblock_1d' module that consist of several "conv1d->BN->relu" repetitions
    
    """
    def __init__(self, num_channels, kernel_size=5):
        super(resblock_1d, self).__init__()
        modules = []
        relu = nn.ReLU(0.2)
        for r in range(2):
            modules.append(
                nn.Sequential(
                    nn.Conv1d(num_channels, num_channels, kernel_size, stride=1, padding=kernel_size//2),  # (batch, width, out_chan)
                    nn.BatchNorm1d(num_channels),
                    relu
                )
            )
        self.block = nn.Sequential(*modules)
        
    def forward(self, inputs):
        outputs = self.block(inputs)
        return outputs


class ResNet1d(nn.Module):
    """ Resudual Network with 1D convolutions for discriminating real vs generated sequences.

    Parameters:
        inputs (tensor):Tensor of size (batch_size, vocab_size, max_seq_len) containing real or generated sequences.
        num_channels (int):Discriminator dimension parameter (in each resnet block).
        vocab_size (int):Size of the first layer input channel.
        seq_len (int):Length of the input sequence.
        num_layers (int):How many repetitions of 'resblock_1d' for discriminator.

    Returns:
        outputs (tensor):Batch of (single) values for real or generated inputs.

    """
    def __init__(self, num_channels=64, vocab_size=4, seq_len=600, num_classes=164, res_layers=2):
        super(ResNet1d, self).__init__()
        
        self.num_channels = num_channels
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.res_layers = res_layers
        self.num_classes = num_classes
                
        self.conv = nn.Sequential(
                nn.Conv1d(vocab_size, num_channels, 1, stride=1, padding=0)  # bottleneck: (batch, width, out_chan)
            )
        resblocks = []
        for i in range(res_layers):
            resblocks.append(resblock_1d(self.num_channels))
        self.resblocks = nn.Sequential(*resblocks)
        self.prediction_layer = nn.Linear(seq_len*num_channels, num_classes)
            
    def forward(self, inputs):
        outputs = self.conv(inputs.reshape(64, self.vocab_size, self.seq_len))
        inputs = outputs
        for i in range(self.res_layers):
            outputs = 1.0*self.resblocks[i](inputs) + inputs  # where resnet idea comes into play!
            inputs = outputs
        outputs = torch.reshape(outputs, [-1, self.seq_len*self.num_channels])
        outputs = self.prediction_layer(outputs)
        print("The shape is:", outputs.shape)
        return outputs

net = ResNet1d() # __init__ here
random_sample = torch.tensor(np.random.randn(64, 4, 1, 600)).float()  # 64 is the batch_size
net(random_sample) # forward here

class ResNet2d(nn.Module):
    """ Basset network to learn models of DNA sequence activity such as accessibility, protein binding, and chromatin state.

    Parameters:
        inputs (tensor): Batch of one-hot-encoded sequences.
        other_arguments: Should be explained later.

    Returns:
        outputs (tensor): Should be explained later.

    """
    def __init__(self, other_arguments=None):
        super(ResNet2d, self).__init__()
        pass

    def forward(self, inputs):
        pass

