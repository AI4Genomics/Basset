# model.py
# --------
# censing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational purposes.# project. You are free to use and extend these projects for educational purposes.

import numpy as np
import torch
import torch.nn as nn

class ResNet2d(nn.Module):
    """ ResNet2d network to learn models of DNA sequence activity such as accessibility, protein binding, and chromatin state.
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
        inputs = torch.reshape(inputs, [-1, 1, self.vocab_size, self.seq_len])
        print(inputs.shape) #See how input will look like after passed to the network and processed

        outputs = self.conv(inputs)
        print(outputs.shape) #Check the output shape of each layer
        """inputs = outputs
        for i in range(self.res_layers):
            outputs = 1.0*self.resblocks[i](inputs) + inputs  # where resnet idea comes into play!
            inputs = outputs
        outputs = torch.reshape(outputs, [-1, self.vocab_size*self.seq_len*self.num_channels])
        outputs = self.prediction_layer(outputs)
        return outputs"""

#note: to check dimensions, do not have to have real data (can be done with any data)
basset_net = Basset(some_arguments) # __init__ here
random_sample = np.random.randn(64, 4, 1, 600) # 64 is the batch_size
basset_net(random_sample) # forward here

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
