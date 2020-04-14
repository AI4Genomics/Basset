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
        pass

    #See how input will look like after passed to the network and processed
    def forward(self, inputs):
        inputs = torch.reshape(inputs, [-1, 1, self.vocab_size, self.seq_len])
        print(inputs.shape) # =======================> YOU SHOULD PAY ATTENTION TO THIS!
        """outputs = self.conv(inputs)
        inputs = outputs
        for i in range(self.res_layers):
            outputs = 1.0*self.resblocks[i](inputs) + inputs  # where resnet idea comes into play!
            inputs = outputs
        outputs = torch.reshape(outputs, [-1, self.vocab_size*self.seq_len*self.num_channels])
        outputs = self.prediction_layer(outputs)
        return outputs"""

    #Check the output shape of each layer
    def forward(self, inputs):
        inputs = torch.reshape(inputs, [-1, 1, self.vocab_size, self.seq_len])
        outputs = self.conv(inputs)
        print(outputs.shape)  # =======================> YOU SHOULD PAY ATTENTION TO THIS!
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
        pass
    def forward(self, inputs):
        pass
