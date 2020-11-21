
import torch
import numpy as np

from torch import nn
from torchvision import transforms as T

# Credit to https://github.com/sksq96/pytorch-vae/blob/master/vae-cnn.ipynb
# For some VAE implementation specifics
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=32):
        return input.view(input.size(0), size, 1, 1)

class VAE(nn.Module):
    def __init__(self, init_weight = 3e-3):
        super(VAE, self).__init__()
        # Define linear layer sizes
        self.z_dimension = 32
        self.hidden_dimension = 32
        self.mean_layer = nn.Linear(self.hidden_dimension, self.z_dimension)
        self.var_layer = nn.Linear(self.hidden_dimension, self.z_dimension)
        self.z_layer = nn.Linear(self.z_dimension, self.hidden_dimension)

        self.mean_layer.weight.data.uniform_(-init_weight, init_weight)
        self.mean_layer.bias.data.uniform_(-init_weight, init_weight)
        self.var_layer.weight.data.uniform_(-init_weight, init_weight)
        self.var_layer.bias.data.uniform_(-init_weight, init_weight)
        self.z_layer.weight.data.uniform_(-init_weight, init_weight)
        self.z_layer.bias.data.uniform_(-init_weight, init_weight)

        # Image input is 3 x 32 x 32
        self.encoder = nn.Sequential(
            # Convert to 16 x 40 x 40
            nn.Conv2d(1, 16, padding = 1, kernel_size = 4, stride = 2, bias = False),
            nn.ReLU(),
            # Convert to 32 x 20 x 20
            nn.Conv2d(16, 32, padding = 1, kernel_size = 4, stride = 2, bias = False),
            nn.ReLU(),
            # Convert to 32 x 10 x 10
            nn.Conv2d(32, 32, padding = 1, kernel_size = 4, stride = 2, bias = False),
            nn.ReLU(),
            # Convert to 32 x 5 x 5
            nn.Conv2d(32, 32, padding = 1, kernel_size = 4, stride = 2, bias = False),
            nn.ReLU(),
            # Convert to 32 x 1 x 1
            nn.Conv2d(32, self.hidden_dimension, padding = 0, kernel_size = 5, stride = 1, bias = False),
            nn.ReLU(),
            Flatten()
        )

        # Input is 1 x 32
        self.decoder = nn.Sequential(
            UnFlatten(),
            # Convert to 32 x 5 x 5
            nn.ConvTranspose2d(self.hidden_dimension, 32, kernel_size = 5, stride = 1, padding = 0),
            nn.ReLU(),
            # Convert to 32 x 10 x 10
            nn.ConvTranspose2d(32, 32, kernel_size = 4, stride = 2, padding = 1),
            nn.ReLU(),
            # Convert to 32 x 20 x 20
            nn.ConvTranspose2d(32, 32, kernel_size = 4, stride = 2, padding = 1),
            nn.ReLU(),
            # Convert to 16 x 40 x 40
            nn.ConvTranspose2d(32, 16, kernel_size = 4, stride = 2, padding = 1),
            nn.ReLU(),
            # Convert to 1 x 80 x 80
            nn.ConvTranspose2d(16, 1, kernel_size = 4, stride = 2, padding = 1),
            nn.Sigmoid(),
        )

    def reparam_and_bottleneck(self, data):
        mean, log_var = self.mean_layer(data.float()), self.var_layer(data.float())
        std = 0.5*log_var.exp()

        return torch.distributions.Normal(mean, std).sample(), mean, log_var

    def encode(self, obsv, batch=True):
        encoded_data = self.encoder(obsv.float())
        z, mean, log_var = self.reparam_and_bottleneck(encoded_data)
        return z, mean, log_var

    def decode(self, encoded_data):
        data = self.z_layer(encoded_data)
        # data = torch.reshape(data, (1, 11, 16))
        decode_data = self.decoder(data[None, ...].float())
        return decode_data

    def forward(self, obsv):
        encoded, mean, log_var = self.encode(obsv)
        # d = self.decode(encoded)

        # We only care about the encoded data.
        return encoded