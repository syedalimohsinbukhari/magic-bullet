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

        # --- CCG Block 1 ---
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.gru1 = nn.GRU(input_size=128, hidden_size=64, num_layers=1, batch_first=True)

        # --- CCG Block 2 ---
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.gru2 = nn.GRU(input_size=64, hidden_size=64, num_layers=1, batch_first=True)

        # --- CCG Block 3 ---
        self.conv5 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.gru3 = nn.GRU(input_size=64, hidden_size=32, num_layers=1, batch_first=True)

        # --- CCG Block 4 ---
        self.conv7 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv8 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.gru4 = nn.GRU(input_size=32, hidden_size=32, num_layers=1, batch_first=True)

        # Final output conv layer
        self.conv_final = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=4096)

    def forward(self, x):
        # Input: [B, 2, L]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.permute(0, 2, 1)  # [B, L, 128]
        x, _ = self.gru1(x)
        x = x.permute(0, 2, 1)  # [B, 64, L]

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.permute(0, 2, 1)  # [B, L, 64]
        x, _ = self.gru2(x)
        x = x.permute(0, 2, 1)  # [B, 64, L]

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = x.permute(0, 2, 1)  # [B, L, 64]
        x, _ = self.gru3(x)
        x = x.permute(0, 2, 1)  # [B, 32, L]

        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = x.permute(0, 2, 1)  # [B, L, 32]
        x, _ = self.gru4(x)
        x = x.permute(0, 2, 1)  # [B, 32, L]

        x = self.conv_final(x)  # [B, 1, L - 4095]
        x = torch.sigmoid(x)  # [B, 1, ...]
        return x.squeeze(1)  # [B, ...]


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
