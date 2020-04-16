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
        modules = []
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
