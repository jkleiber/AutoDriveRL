
import torch
import numpy as np
import os
import random

from torch import nn
from torch.nn import functional as F
from torch import optim

from agent import Agent

class SoftActorCriticAgent(Agent):
    def __init__(self, batch_size=64, lr=1e-3, tau = 0.01, gamma = 0.99, eps = 1e-6, max_replay_size = 100000, save_path = 'soft_actor_critic/'):
        # Start with the replay pool empty
        self.replay_pool = []
        self.max_replay_size = max_replay_size
        self.num_experiences = 0

        # Initialize the target and Q function networks
        self.policy_network = PolicyNetwork()
        self.target_net_1 = SoftQFunctionNetwork()
        self.target_net_2 = SoftQFunctionNetwork()
        self.q_network_1 = SoftQFunctionNetwork()
        self.q_network_2 = SoftQFunctionNetwork()

        # Loss Functions
        self.policy_loss = nn.MSELoss()
        self.q_loss_1 = nn.MSELoss()
        self.q_loss_2 = nn.MSELoss()

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.q_optimizer_1 = optim.Adam(self.q_network_1.parameters(), lr=lr)
        self.q_optimizer_2 = optim.Adam(self.q_network_2.parameters(), lr=lr)

        # Temperature and Distributions
        self.alpha = 1
        self.z_norm = torch.distributions.Normal(0, 1)

        # Hyperparameters
        self.lr = lr
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.epsilon = eps

        # Save path
        self.save_path = os.getcwd() + '/' + save_path

        self.cuda_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda_available else "cpu")

    def act(self, obsv):
        # Convert the observed state to a torch tensor
        input_data = np.copy(obsv)
        input_data.setflags(write=1)
        obsv_tensor = torch.Tensor(input_data)

        # Permute the tensor to put channels as first dimension.
        obsv_tensor = obsv_tensor.permute(2, 0, 1)

        # Find the log of the action recommended by the policy network.
        policy_dist = self.policy_network(obsv_tensor[None, ...])

        # Find the action using the reparameterization trick.
        mean = policy_dist[0]
        std = policy_dist[1]
        action = torch.tanh(mean + std * self.z_norm.sample().to(self.device))
        action = action.detach().numpy()

        return action

    def update(self, old_obsv, action, reward, new_obsv, is_crashed):
        # Add the most recent experience to the replay pool.
        old_obsv_copy = np.copy(old_obsv)
        old_obsv_copy.setflags(write=1)
        # old_obsv_t = torch.Tensor(old_obsv_copy).permute(2, 0, 1)
        new_obsv_copy = np.copy(new_obsv)
        new_obsv_copy.setflags(write=1)
        # new_obsv_t = torch.Tensor(new_obsv_copy).permute(2, 0, 1)
        self.update_replay_pool(old_obsv_copy, action, reward, new_obsv_copy, is_crashed)

        # Get a sample of experiences from the replay pool.
        replay_batch = self.sample_replay_pool()

        if len(replay_batch) == 0:
            return

        # Unpack the data from the replay batch.
        states = np.array([s for s, a, r, new_s, c in replay_batch])
        actions = [a for s, a, r, new_s, c in replay_batch]
        rewards = [r for s, a, r, new_s, c in replay_batch]
        new_states = [new_s for s, a, r, new_s, c in replay_batch]
        crashes = [c for s, a, r, new_s, c in replay_batch]

        # Make tensors and send them to our device (CPU / GPU).
        state_t = torch.from_numpy(states).permute(0, 3, 1, 2).to(self.device)
        action_t = torch.Tensor(actions).to(self.device)
        reward_t = torch.Tensor(rewards).unsqueeze(1).to(self.device)
        new_state_t = torch.Tensor(new_states).permute(0, 3, 1, 2).to(self.device)
        crash_t = torch.Tensor(crashes).unsqueeze(1).to(self.device)

        # Sample policies from the batch of states
        means = []
        stds = []
        new_action_t = torch.Tensor()
        log_prob_t = torch.Tensor()
        for s in state_t:
            # Find the predicted action
            pred = self.policy_network(s[None, ...])
            mean = pred[0]
            std = pred[1]
            action = torch.tanh(mean + std * self.z_norm.sample().to(self.device)).unsqueeze(0)
            log_prob = torch.distributions.Normal(mean, std).log_prob(action) - torch.log(1 - action.pow(2) + self.epsilon)

            # Save the predictions
            means.append(mean)
            stds.append(std)
            new_action_t = torch.cat([new_action_t, action],0)
            log_prob_t = torch.cat([log_prob_t, log_prob],0)

        ### Q function optimization
        pred_q_1 = self.q_network_1(state_t, action_t)
        pred_q_2 = self.q_network_2(state_t, action_t)

        # Find the target Q value
        target_value = torch.min(self.target_net_1(state_t, action_t), self.target_net_2(state_t, action_t))
        target_q = reward_t + (1-crash_t) * self.gamma * target_value

        # Loss functions
        q_value_loss_1 = self.q_loss_1(pred_q_1, target_q.detach())
        q_value_loss_2 = self.q_loss_2(pred_q_2, target_q.detach())

        # Optimize
        self.q_optimizer_1.zero_grad()
        q_value_loss_1.backward()
        self.q_optimizer_1.step()

        self.q_optimizer_2.zero_grad()
        q_value_loss_2.backward()
        self.q_optimizer_2.step()

        ### Target Function optimization
        pred_q = torch.min(self.q_network_1(state_t, new_action_t), self.q_network_2(state_t, new_action_t))

        self.update_target_function(self.q_network_1, self.target_net_1)
        self.update_target_function(self.q_network_2, self.target_net_2)

        ### Policy optimization.
        # Compute loss by comparing to the Q function output
        policy_q_loss = (log_prob_t - pred_q).mean()

        self.policy_optimizer.zero_grad()
        policy_q_loss.backward()
        self.policy_optimizer.step()

    def update_target_function(self, q_net, target_net):
        """ Modified from SAC RLKit Implementation """
        for target_param, param in zip(target_net.parameters(), q_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

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

    def save_weights(self):
        torch.save(self.policy_network.state_dict(), self.save_path + 'sac_policy.net')
        torch.save(self.q_network_1.state_dict(), self.save_path + 'sac_q1.net')
        torch.save(self.q_network_2.state_dict(), self.save_path + 'sac_q2.net')
        torch.save(self.target_net_1.state_dict(), self.save_path + 'sac_target1.net')
        torch.save(self.target_net_2.state_dict(), self.save_path + 'sac_target2.net')

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
        self.linear1 = nn.Linear(266, 266)
        self.linear2 = nn.Linear(266, 266)
        self.linear3 = nn.Linear(266, 2)

        self.mean_layer = nn.Linear(266, 2)

    def forward(self, obsv):
        # Convolutional Layers
        c1 = F.relu(self.conv1(obsv.float()))
        c2 = F.relu(self.conv2(c1))
        c3 = F.relu(self.conv3(c2))
        c4 = F.relu(self.conv4(c3))

        # Flatten convolutional output for linear layers
        linear_input = torch.flatten(c4)
        x1 = F.relu(self.linear1(linear_input))
        x2 = F.relu(self.linear2(x1))
        x3 = F.relu(self.linear3(x2))

        # Create distribution
        mean = self.mean_layer(x2)
        std = torch.exp(x3)

        return (mean, std)

class SoftQFunctionNetwork(nn.Module):
    def __init__(self):
        super(SoftQFunctionNetwork, self).__init__()
        input_size = (120 * 160 * 3) + 2
        self.linear1 = nn.Linear(input_size, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 1)

    def forward(self, state, action):
        flat_state = torch.flatten(state, start_dim = 1)
        q_input = torch.cat([flat_state, action], 1)
        q1 = F.relu(self.linear1(q_input))
        q2 = F.relu(self.linear2(q1))
        q3 = F.relu(self.linear3(q2))

        return q3
