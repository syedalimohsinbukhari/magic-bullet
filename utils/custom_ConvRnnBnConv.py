"""
Define the fully convolutional neural net (FCNN) model to be trained.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np

# -----------------------------------------------------------------------------
# MODEL DEFINITIONS
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=512, kernel_size=1)
        self.rnn1 = nn.GRU(input_size=512, hidden_size=256, batch_first=True)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv7 = nn.Conv1d(in_channels=256, out_channels=1, kernel_size=4096)

    def forward(self, x):
            # Convolutional + Activation + Optional Pooling
        x = F.relu(self.conv1(x))
        x = x.permute(0, 2, 1)
        x, _ = self.rnn1(x)               # [B, L, 256]
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(x))
        x = self.conv7(x)
        x = torch.sigmoid(x)  # Binary classification

        return x


# -----------------------------------------------------------------------------
# MAIN CODE (= BASIC TESTING ZONE)
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # Instantiate the default model
    print('Instantiating model...', end=' ', flush=True)
    model = FCNN()
    print('Done!', flush=True)

    # Compute the number of trainable parameters
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_parameters = sum([np.prod(p.size()) for p in parameters])
    print('Number of trainable parameters:', n_parameters, '\n')

    # Create some dummy input
    print('Creating random input...', end=' ', flush=True)
    data = torch.randn((1, 2, 8 * 2048))
    print('Done!', flush=True)
    print('Input shape:', data.shape, '\n')

    # Compute the forward pass through the model
    print('Computing forward pass...', end=' ', flush=True)
    output = model.forward(data)
    print('Done!', flush=True)
    print('Output shape:', output.shape)
