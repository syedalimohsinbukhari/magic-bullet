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

        # Block 1: Conv → Conv → LSTM
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.lstm1 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)

        # Block 2: Conv → Conv → LSTM
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, batch_first=True)

        # Final Conv layer
        self.conv_final = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=4096)

    def forward(self, x):
        # Input: [B, 2, L]
        x = F.relu(self.conv1(x))  # [B, 64, L]
        x = F.relu(self.conv2(x))  # [B, 128, L]

        # Prepare for LSTM1: [B, 128, L] → [B, L, 128]
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)  # [B, L, 64]
        x = x.permute(0, 2, 1)  # [B, 64, L]

        x = F.relu(self.conv3(x))  # [B, 128, L]
        x = F.relu(self.conv4(x))  # [B, 64, L]

        # LSTM2
        x = x.permute(0, 2, 1)  # [B, L, 64]
        x, _ = self.lstm2(x)  # [B, L, 32]
        x = x.permute(0, 2, 1)  # [B, 32, L]

        # Final conv
        x = self.conv_final(x)  # [B, 1, L]
        x = torch.sigmoid(x)  # [B, 1, L]
        return x.squeeze(1)  # [B, L]


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
