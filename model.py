# model.py
# --------
# censing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational purposes.# project. You are free to use and extend these projects for educational purposes.

import numpy as np
import torch
import torch.nn as nn

class basset(nn.Module):
    """ Basset network to learn models of DNA sequence activity such as accessibility, protein binding, and chromatin state.

    Parameters:
        inputs (tensor): Batch of one-hot-encoded sequences.
	other_arguments: Should be explained later.

    Returns:
        outputs (tensor): Should be explained later.

    """
    def __init__(self, other_arguments=None):
        super(basset, self).__init__()
	pass

    def forward(self, inputs):
        pass
