
import torch
import numpy as np
import random

from torch import nn
from torch.nn import functional as F
from torch import optim

from agent import Agent

class SoftActorCriticAgent(Agent):
    def __init__(self, batch_size=64, lr=1e-4, max_replay_size = 100000):
        # Start with the replay pool empty
        self.replay_pool = []
        self.max_replay_size = max_replay_size
        self.num_experiences = 0

        # Initialize the target and Q function networks
        self.policy_network = PolicyNetwork()
        self.q_network_1 = SoftQFunctionNetwork()
        self.q_network_2 = SoftQFunctionNetwork()

        # Loss Functions
        # self.policy_loss = F.MSELoss()

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.q_optimizer_1 = optim.Adam(self.q_network_1.parameters(), lr=lr)
        self.q_optimizer_2 = optim.Adam(self.q_network_2.parameters(), lr=lr)

        # Temperature
        self.alpha = 1

        # Hyperparameters
        self.lr = lr
        self.batch_size = batch_size

        self.cuda_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda_available else "cpu")

    def act(self, obsv):
        # Convert the observed state to a torch tensor
        input_data = np.copy(obsv)
        input_data.setflags(write=1)
        obsv_tensor = torch.Tensor(input_data)

        # Permute the tensor to put channels as first dimension
        obsv_tensor = obsv_tensor.permute(2, 0, 1)

        # Find the log of the action recommended by the policy network.
        log_action = self.policy_network(obsv_tensor[None, ...]).detach().numpy()

        # Find the action using the reparameterization trick.
        action = self.alpha * log_action
        print(action)

        return action

    def update(self, old_obsv, action, reward, new_obsv, is_crashed):
        # Add the most recent experience to the replay pool
        self.update_replay_pool(old_obsv, action, reward, new_obsv, is_crashed)

        # Get a sample of experiences from the replay pool.
        replay_batch = self.sample_replay_pool()

        if len(replay_batch) > 0:
            pass


    def sample_replay_pool(self):
        # If the agent doesn't have enough experience yet, return nothing.
        if len(self.replay_pool) < self.batch_size:
            return []

        # Sample from past experiences randomly.
        batch = random.sample(self.replay_pool, self.batch_size)

        return batch

    def update_replay_pool(self, old_obsv, action, reward, new_obsv, is_crashed):
        experience = (old_obsv, action, reward, new_obsv, is_crashed)
        if len(self.replay_pool) < self.max_replay_size:
            self.replay_pool.append(experience)
        else:
            self.replay_pool[self.num_experiences % self.max_replay_size] = experience
        self.num_experiences += 1


class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        # Image input is 3 x 120 x 160
        # Convert to 16 x 60 x 80
        self.conv1 = nn.Conv2d(3, 16, padding = 1, kernel_size = 4, stride = 2)
        # Convert to 32 x 30 x 40
        self.conv2 = nn.Conv2d(16, 32, padding = 1, kernel_size = 4, stride = 2)
        # Convert to 64 x 15 x 20
        self.conv3 = nn.Conv2d(32, 64, padding = 1, kernel_size = 4, stride = 2)
        # Convert to 1 x 14 x 19
        self.conv4 = nn.Conv2d(64, 1, padding = 1, kernel_size = 4, stride = 1)

        # Linear layers to get to output of 2x1
        # Note: make sure to flatten conv2d output
        self.linear1 = nn.Linear(266, 64)
        self.linear2 = nn.Linear(64, 16)
        self.linear3 = nn.Linear(16, 2)

    def forward(self, obsv):
        # Convolutional Layers
        c1 = F.relu(self.conv1(obsv))
        c2 = F.relu(self.conv2(c1))
        c3 = F.relu(self.conv3(c2))
        c4 = F.relu(self.conv4(c3))

        # Flatten convolutional output for linear layers
        linear_input = torch.flatten(c4)
        x1 = F.relu(self.linear1(linear_input))
        x2 = F.relu(self.linear2(x1))
        x3 = F.relu(self.linear3(x2))

        return x3

class SoftQFunctionNetwork(nn.Module):
    def __init__(self):
        super(SoftQFunctionNetwork, self).__init__()
        self.linear1 = nn.Linear(50, 2)

    def forward(self, obsv):
        pass