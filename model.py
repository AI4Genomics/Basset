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

        self.num_channels = num_channels
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.layers = layers

        self.initial_layer = nn.Linear(latent_dim, vocab_size * seq_len * num_channels)
        blocks = []
        for i in range(layers):
            blocks.append(
                self.blocks = nn.Sequential(*blocks)
                self.conv = nn.Sequential(
                    nn.Conv2d(num_channels, 1, (3, 1), stride=(1, 1), padding=(3 // 2, 0)), # bottleneck: (batch, width, out_chan)
                    relu
                )
            )
    def forward(self, inputs):
        outputs = self.initial_layer(inputs)
        outputs = torch.reshape(outputs, [-1, self.num_channels, self.vocab_size, self.seq_len])
        inputs = outputs
        for i in range(self.layers):
            outputs = 1.0 * self.blocks[i](inputs) + inputs  # where resnet idea comes into play!
            inputs = outputs
        outputs = self.conv(outputs)
        outputs = torch.reshape(outputs, [-1, self.vocab_size, self.seq_len])
        return outputs
