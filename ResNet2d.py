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
        num_channels (int):Generator dimension parameter (in each resnet block).
           latent_dim (int):Size of latent space (i.e., the first layer dimension).
           vocab_size (int):Size of the last layer output channel.
           seq_len (int):Length of the output sequence.
           num_layers (int):How many repetitions of 'resblock_2d' for generator.

        forward(input) successively applies the input data to the different layers defined in __init__
    Returns:
        outputs (tensor):Batch of (single) values for real or generated inputs.

    """

    def __init__(self, num_channels, latent_dim, vocab_size, seq_len, layers=5):
        super(Basset, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(4, 16, (2,2), stride=(1,1), padding=(3//2, 0)),
            nn.ReLU,
            nn.MaxPool2d(2))

        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, (2,2), stride=(1,1), padding=(3//2, 0)),
            nn.ReLU,
            nn.MaxPool2d(2)

        self.fc = nn.Linear(vocab_size*seq_len*32, 10)
        )

    def forward(self, inputs):
        outputs = self.block1(inputs)
        outputs - self.block2(outputs)

        #Flatten the output of block2
        outputs = outputs.view(outputs.size(0), -1)

        outputs = self.fc(outputs)
        return outputs
