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
	    `torch.nn.Conv2d(in_channels, out_channels, kernel_size)` applies a 2D convolution on the input channels.
        `torch.nn.MaxPool2d(kernel_size)` applies 2D max pooling on the input channels
        `torch.nn.Relu()` applies an elementwise Relu activation: Relu(x) = max(0, x).
        `torch.nn.Linear(in_features, out_features)` applies a linear transformation on its input: y = Ax + b.
        `torch.nn.Softmax(dim)` applies a softmax activation to an n-dimensional tensor (normalizes the exponentiated entries)
        `torch.nn.sequential` a sequential container in which to add modules in the order in which they will be constructed.

        forward(input) successively applies the input data to the different layers defined in __init__
    Returns:
        outputs (tensor): Should be explained later.

    """
    def __init__(self, other_arguments=None):
        super(Basset, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, inputs):
        out = self.block1(inputs)

        out = self.block2(out)

        # Flatten the output of block2
        out = out.view(out.size(0), -1)

        out = self.fc(out)

        return out

    model = Basset()
