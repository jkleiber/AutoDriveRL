
import torch
import numpy as np
import os
import random

from math import log
from torch import nn
from torch.nn import functional as F
from torch import optim

from agent import Agent
from vae import VAE

class ACER(Agent):
    MAX_THROTTLE = 0.5
    MIN_THROTTLE = 0.0
    STEER_CONST = 1

    def __init__(self, vae_lr = 1e-4, sdn_lr = 1e-3, policy_lr = 1e-4):
        # See if we can do GPU training
        self.cuda_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda_available else "cpu")

        # Start with the replay pool empty
        self.replay_pool = []
        self.max_replay_size = max_replay_size
        self.num_experiences = 0

        # VAE (Variational Autoencoder)
        self.vae_net = VAE()
        self.vae_optimizer = optim.Adam(self.vae_net.parameters(), lr = vae_lr)

        # Stochastic Dueling Network (SDN)
        self.sdn_net = SDN()
        self.sdn_optimizer = optim.Adam(self.sdn_net.parameters(), lr = sdn_lr)

        # Policy Network
        self.policy_network = ACERPolicy()
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr = policy_lr)

    def trim_image(self, img_t):
        trimmed_t = img_t[:, 40:, :]
        return trimmed_t

    def train_vae(self, batch):
        total_loss = 0
        for img in batch:
            # Encode the batch
            img = self.trim_image(img)
            encoded_img, mean, log_var = self.vae_net.encode(img[None, ...].float())

            # Decode the encoded batch
            decoded_img = self.vae_net.decode(encoded_img[None, ...].float())

            # Compute the loss and KL divergence
            img = img.unsqueeze(0)
            vae_loss = F.mse_loss(img.float(), decoded_img.float())
            kl_divergence = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
            loss = vae_loss + kl_divergence
            total_loss += loss

            # Optimize
            self.vae_optimizer.zero_grad()
            loss.backward()
            self.vae_optimizer.step()
        print(f'VAE Loss: {total_loss.mean()}')

    def act(self, obsv, cur_action):
        # Convert the observed state to a torch tensor
        input_data = np.copy(obsv)
        input_data.setflags(write=1)
        obsv_tensor = torch.Tensor(input_data)

        # Permute the tensor to put channels as first dimension.
        # obsv_tensor = obsv_tensor.permute(2, 0, 1)
        obsv_tensor = obsv_tensor.unsqueeze(0)

        # Put the observation through a VAE to get an encoding
        trimmed_obsv = img = self.trim_image(obsv_tensor)
        latent_t = self.vae_net(trimmed_obsv[None, ...])

        # Find the action recommended by the policy network.
        mean, std, value = self.policy_network(latent_t)

        # Decode action from policy
        raw_action = self.decode_action(mean, std)

        # Scale action to be within control limits
        action = raw_action
        action[1] = ((self.MAX_THROTTLE - self.MIN_THROTTLE) / 2.0) * action[1] + ((self.MAX_THROTTLE - self.MIN_THROTTLE) / 2.0)
        action[0] = cur_action[0] + self.STEER_CONST*(action[0] - cur_action[0])

        return action, raw_action

    def update(self):
        pass

    def decode_action(self, mean, std):
        action = torch.distributions.Normal(mean, std).sample()

        return action

    def add_experience(self, old_obsv, action, reward, new_obsv, is_crashed):
        # Add the most recent experience to the replay pool.
        old_obsv_copy = np.copy(old_obsv)
        old_obsv_copy.setflags(write=1)
        new_obsv_copy = np.copy(new_obsv)
        new_obsv_copy.setflags(write=1)
        self.update_replay_pool(old_obsv_copy, action, reward, new_obsv_copy, is_crashed)
    
    def update_replay_pool(self, old_obsv, action, reward, new_obsv, is_crashed):
        experience = (old_obsv, action, reward, new_obsv, is_crashed)
        if len(self.replay_pool) < self.max_replay_size:
            self.replay_pool.append(experience)
        else:
            self.replay_pool[self.num_experiences % self.max_replay_size] = experience
        self.num_experiences += 1

    def save_weights(self):
        # torch.save(self.policy_network.state_dict(), self.save_path + 'sac_policy.net')
        # torch.save(self.q_network_1.state_dict(), self.save_path + 'sac_q1.net')
        # torch.save(self.q_network_2.state_dict(), self.save_path + 'sac_q2.net')
        # torch.save(self.target_net_1.state_dict(), self.save_path + 'sac_target1.net')
        # torch.save(self.target_net_2.state_dict(), self.save_path + 'sac_target2.net')
        # torch.save(self.vae_net.state_dict(), self.save_path + 'sac_vae.net')
        # torch.save(self.log_alpha, self.save_path + 'log_alpha.pt')

    def init_with_saved_weights(self):
        # self.policy_network.load_state_dict(torch.load(self.save_path + 'sac_policy.net'))
        # self.q_network_1.load_state_dict(torch.load(self.save_path + 'sac_q1.net'))
        # self.q_network_2.load_state_dict(torch.load(self.save_path + 'sac_q2.net'))
        # self.target_net_1.load_state_dict(torch.load(self.save_path + 'sac_target1.net'))
        # self.target_net_2.load_state_dict(torch.load(self.save_path + 'sac_target2.net'))
        # self.vae_net.load_state_dict(torch.load(self.save_path + 'sac_vae.net'))
        # self.log_alpha = torch.load(self.save_path + 'log_alpha.pt')

        # self.policy_network.eval()
        # self.q_network_1.eval()
        # self.q_network_2.eval()
        # self.target_net_1.eval()
        # self.target_net_2.eval()
        # self.vae_net.eval()
        # self.alpha = self.log_alpha.exp()

# Acknowledgement to https://github.com/dchetelat/acer/
# for some implementation details
def ACERPolicy(nn.Module):
    def __init__(self, input_size = 32, hidden_size = 64):
        super(ACERPolicy, self).__init__()

        # Initial Layers
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)

        # Mean and std layers
        self.mean_layer = nn.Linear(hidden_size, 2)
        self.log_layer = nn.Linear(hidden_size, 2)

        # Value layer
        self.value_layer = nn.Linear(hidden_size, 1)

    def forward(self, input_t):
        h = F.relu(self.input_layer(input_t))
        h = F.relu(self.hidden_layer(h))

        # Find distribution
        mean = self.mean_layer(h)
        log_std = self.log_layer(h)
        std = log_std.exp()

        # Find value
        value = self.value_layer(h)

        return mean, std, value


def SDN(nn.Module):
    def __init__(self):
        super(SDN, self).__init__()


