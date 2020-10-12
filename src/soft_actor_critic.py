
import torch
import numpy as np

from torch import nn
from torch.nn import functional as F
from torch import optim

from agent import Agent

class SoftActorCriticAgent(Agent):
    def __init__(self, obsv_dim, act_dim, batch_size=64, lr=1e-4, max_replay_size = 100000):
        # Start with the replay pool empty
        self.replay_pool = []
        self.max_replay_size = max_replay_size
        self.num_experiences = 0

        # Initialize the target and Q function networks
        num_obsv_inputs = obsv_dim[0] * obsv_dim[1]
        num_action_outputs = act_dim[0] * act_dim[1]
        self.policy_network = PolicyNetwork(num_obsv_inputs, out_dim)
        self.q_network_1 = SoftQFunctionNetwork(num_action_outputs + num_obsv_inputs, out_dim)
        self.q_network_2 = SoftQFunctionNetwork(num_action_outputs + num_obsv_inputs, out_dim)

        # Loss Functions
        # self.policy_loss = F.MSELoss()

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.q_optimizer_1 = optim.Adam(self.q_network_1.parameters(), lr=lr)
        self.q_optimizer_2 = optim.Adam(self.q_network_2.parameters(), lr=lr)

        # Temperature
        self.alpha = 0

        # Hyperparameters
        self.lr = lr
        self.batch_size = batch_size

        self.cuda_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda_available else "cpu")

    def act(self, obsv):
        # Convert the observed state to a torch tensor
        obsv_tensor = torch.Tensor(obsv)

        # Find the log of the action recommended by the policy network.
        log_action = self.policy_network(obsv_tensor)

        # Find the action using the reparameterization trick.
        action = self.alpha * log_action

        return action

    def update(self, old_obsv, action, reward, new_obsv, is_crashed):
        # Add the most recent experience to the replay pool
        update_replay_pool(old_obsv, action, reward, new_obsv, is_crashed)

        # Get a sample of experiences from the replay pool.
        replay_batch = sample_replay_pool()

        if len(replay_batch) > 0:
            pass


    def sample_replay_pool(self):
        # If the agent doesn't have enough experience yet, return nothing.
        if len(self.replay_pool) < self.batch_size:
            return []

        # Sample from past experiences randomly.
        batch = random.sample(self.replay_pool, self.batch_size)

    def update_replay_pool(self, old_obsv, action, reward, new_obsv, is_crashed):
        experience = (old_obsv, action, reward, new_obsv, is_crashed)
        if len(self.replay_pool) < self.max_replay_size:
            self.replay_pool.append(experience)
        else:
            self.replay_pool[self.num_experiences % self.max_replay_size] = experience
        self.num_experiences += 1


class PolicyNetwork(nn.Module):
    def __init__(self, obsv_dim, out_dim):
        self.conv1 = nn.Conv2d(3, 3, padding = 1, kernel_size = 3)
        self.linear1 = nn.Linear(obsv_dim, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 2)

    def forward(self, obsv):
        x1 = nn.ReLU(self.linear1(obsv))
        x2 = nn.ReLU(self.linear2(x1))
        x3 = nn.ReLU(self.linear3(x2))

        return x3

class SoftQFunctionNetwork(nn.Module):
    def __init__(self, obsv_dim, out_dim):
        self.linear1 = nn.Linear()

    def forward(self, obsv):
        pass