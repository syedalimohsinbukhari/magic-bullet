import torch
import torch.nn as nn

class RNNModel(nn.Module):

    def __init__(self,
                 n_channels: int = 512,
                 n_rnn_layers: int = 12):

        super(RNNModel, self).__init__()

        # ---------------------------------------------------------------------
        # Store constructor arguments
        # ---------------------------------------------------------------------

        self.n_channels = n_channels
        self.n_rnn_layers = n_rnn_layers

        # ---------------------------------------------------------------------
        # Define the model's layers
        # ---------------------------------------------------------------------

        # Input layer (Conv1D remains unchanged)
        self.input_layer = nn.Conv1d(in_channels=2,
                                     out_channels=self.n_channels,
                                     kernel_size=1)

        # List to store RNN layers
        self.rnn_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()

        # Define RNN layers (GRU for efficiency)
        for i in range(self.n_rnn_layers):
            rnn_layer = nn.GRU(input_size=self.n_channels,
                               hidden_size=self.n_channels,
                               batch_first=True)

            self.rnn_layers.append(rnn_layer)

            # Add BatchNormalization after every 3rd layer
            if (i + 1) % 3 == 0:
                self.batch_norm_layers.append(nn.BatchNorm1d(self.n_channels))

        # Output layer (1D Convolution to map to a single output channel)
        self.output_layer = nn.Conv1d(in_channels=self.n_channels,
                                      out_channels=1,
                                      kernel_size=1)

        # ---------------------------------------------------------------------
        # Initialize the weights of the model
        # ---------------------------------------------------------------------

        # Initialize the weight and bias of the input & output layers
        nn.init.kaiming_normal_(self.input_layer.weight)
        nn.init.kaiming_normal_(self.output_layer.weight)
        nn.init.constant_(self.input_layer.bias, 0.001)
        nn.init.constant_(self.output_layer.bias, 0.001)

        # Initialize BatchNorm layers
        for bn_layer in self.batch_norm_layers:
            nn.init.constant_(bn_layer.weight, 1)
            nn.init.constant_(bn_layer.bias, 0)

    # -------------------------------------------------------------------------

    def forward(self, x):
        # Input layer transformation (Conv1D)
        x = self.input_layer(x)

        # Reshape for RNN input: (batch, channels, sequence) → (batch, sequence, channels)
        x = x.permute(0, 2, 1)

        # Pass through RNN layers
        for i, rnn_layer in enumerate(self.rnn_layers):
            x, _ = rnn_layer(x)  # GRU layer forward pass
            
            # Apply batch normalization after every 3rd layer
            if (i + 1) % 3 == 0:
                x = self.batch_norm_layers[(i + 1) // 3 - 1](x.permute(0, 2, 1)).permute(0, 2, 1) 

            x = torch.relu(x)  # Activation function

        # Reshape back to (batch, channels, sequence) for Conv1D output
        x = x.permute(0, 2, 1)

        # Output layer (Conv1D)
        x = self.output_layer(x)
        x = torch.sigmoid(x)  # Sigmoid activation for binary classification

        return x
