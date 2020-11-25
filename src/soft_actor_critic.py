
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

class SoftActorCriticAgent(Agent):
    MAX_THROTTLE = 0.25
    MIN_THROTTLE = 0.0
    STEER_CONST = 1

    def __init__(self, batch_size=64, alpha_lr=3e-4, soft_q_lr = 3e-4, policy_lr = 3e-4, vae_lr = 3e-4, tau = 0.005, gamma = 0.99,
                 eps = 1e-6, alpha = 1.0, num_updates = 1, max_replay_size = 50000, save_path = 'soft_actor_critic/'):
        # See if we can do GPU training
        self.cuda_available = torch.cuda.is_available()
        self.device = torch.device("cpu") # Need to figure out cuda
        # self.device = torch.device("cuda" if self.cuda_available else "cpu")

        # Start with the replay pool empty
        self.replay_pool = []
        self.max_replay_size = max_replay_size
        self.num_experiences = 0

        # Initialize the target and Q function networks
        self.policy_network = PolicyNetwork(eps = eps)
        self.target_net_1 = SoftQFunctionNetwork()
        self.target_net_2 = SoftQFunctionNetwork()
        self.q_network_1 = SoftQFunctionNetwork()
        self.q_network_2 = SoftQFunctionNetwork()

        # Temperature
        self.alpha = alpha
        self.entropy_target = -torch.prod(torch.Tensor((2,1)).to(self.device)).item()
        self.log_alpha = torch.tensor([log(alpha)], requires_grad=True, device = self.device)
        # self.z_norm = torch.distributions.Normal(0, 1)

        # Loss Functions
        self.policy_loss = nn.MSELoss()
        self.q_loss_1 = nn.MSELoss()
        self.q_loss_2 = nn.MSELoss()

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=policy_lr)
        self.q_optimizer_1 = optim.Adam(self.q_network_1.parameters(), lr=soft_q_lr)
        self.q_optimizer_2 = optim.Adam(self.q_network_2.parameters(), lr=soft_q_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        # VAE
        self.vae_net = VAE()
        self.vae_optimizer = optim.Adam(self.vae_net.parameters(), lr = vae_lr)

        # Update step tracker
        self.update_step = 0
        self.update_interval = 0
        self.num_updates = num_updates

        # Hyperparameters
        self.policy_lr = policy_lr
        self.soft_q_lr = soft_q_lr
        self.alpha_lr = alpha_lr
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma

        # Save path
        self.save_path = os.getcwd() + '/' + save_path

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

    def act(self, obsv, cur_action, eval_mode):
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

        # Find the log of the action recommended by the policy network.
        mean, std = self.policy_network(latent_t)

        # Decode the action from the given distribution estimate
        if eval_mode is False:
            raw_action = self.decode_action(mean, std).flatten()
            raw_action = raw_action.detach().numpy()
        else:
            raw_action = mean.detach().numpy()

        # scale action to be between 0 and 1
        action = np.array([raw_action[0], 0.05])
        # action[1] = 0.05#((self.MAX_THROTTLE - self.MIN_THROTTLE) / 2.0) * action[1] + ((self.MAX_THROTTLE - self.MIN_THROTTLE) / 2.0)
        # action[0] = cur_action[0] + self.STEER_CONST*(action[0] - cur_action[0])

        return action, raw_action

    def decode_action(self, mean, std):
        z_sample = torch.distributions.Normal(mean, std).sample()
        action = torch.tanh(z_sample)

        return action

    def add_experience(self, old_obsv, action, reward, new_obsv, is_crashed):
        # Add the most recent experience to the replay pool.
        old_obsv_copy = np.copy(old_obsv)
        old_obsv_copy.setflags(write=1)
        new_obsv_copy = np.copy(new_obsv)
        new_obsv_copy.setflags(write=1)
        self.update_replay_pool(old_obsv_copy, action, reward, new_obsv_copy, is_crashed)

    def update(self):
        for i in range(self.num_updates):
            # Get a sample of experiences from the replay pool.
            replay_batch = self.sample_replay_pool()

            if len(replay_batch) == 0:
                return

            # Unpack the data from the replay batch.
            states = np.array([s for s, a, r, new_s, c in replay_batch])
            actions = [a for s, a, r, new_s, c in replay_batch]
            rewards = [r for s, a, r, new_s, c in replay_batch]
            new_states = np.array([new_s for s, a, r, new_s, c in replay_batch])
            crashes = [c for s, a, r, new_s, c in replay_batch]

            # Make tensors and send them to our device (CPU / GPU).
            # state_t = torch.from_numpy(states).permute(0, 3, 1, 2).to(self.device)
            state_t = torch.from_numpy(states).unsqueeze(1).to(self.device)
            action_t = torch.FloatTensor(actions).to(self.device)
            reward_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
            # new_state_t = torch.Tensor(new_states).permute(0, 3, 1, 2).to(self.device)
            new_state_t = torch.from_numpy(new_states).unsqueeze(1).to(self.device)
            crash_t = torch.FloatTensor(crashes).unsqueeze(1).to(self.device)

            ### Q function optimization
            pred_q_1 = self.q_network_1(state_t, action_t)
            pred_q_2 = self.q_network_2(state_t, action_t)

            # Find the target Q value
            target_value = torch.min(self.target_net_1(new_state_t, action_t), self.target_net_2(new_state_t, action_t))
            target_q = reward_t + (1-crash_t) * self.gamma * target_value

            # Loss functions
            q_value_loss_1 = self.q_loss_1(pred_q_1, target_q.detach())
            q_value_loss_2 = self.q_loss_2(pred_q_2, target_q.detach())

            print(f'Q1 Loss: {q_value_loss_1}')
            print(f'Q2 Loss: {q_value_loss_2}')

            # Optimize
            self.q_optimizer_1.zero_grad()
            q_value_loss_1.backward()
            self.q_optimizer_1.step()

            self.q_optimizer_2.zero_grad()
            q_value_loss_2.backward()
            self.q_optimizer_2.step()

            # Train the VAE
            # Uncomment to train the VAE
            # self.train_vae(state_t)

            # Create a batch of latent_tensors
            latent_t = self.vae_net(state_t).detach()

            # Sample policies from VAE generated tensors
            new_action_t, log_pi_t = self.policy_network.batch(latent_t)

            ### Perform target and policy optimization less often than Q function optimization
            if self.update_step >= self.update_interval:
                ### Target Function optimization
                pred_q = torch.min(
                    self.q_network_1(state_t, new_action_t),
                    self.q_network_2(state_t, new_action_t)
                    )

                self.update_target_function(self.q_network_1, self.target_net_1)
                self.update_target_function(self.q_network_2, self.target_net_2)

                ### Policy optimization.
                # Compute loss by comparing to the Q function output
                # policy_q_loss = F.mse_loss((self.alpha * log_pi_t), pred_q)
                policy_q_loss = self.policy_loss((self.alpha * log_pi_t), pred_q)

                print(f'Policy Loss: {policy_q_loss}')

                self.policy_optimizer.zero_grad()
                policy_q_loss.backward()
                self.policy_optimizer.step()

                # Reset the update stepper (will increase to 0 at end of update function)
                self.update_step = -1

            # Update the temperature
            alpha_loss = (self.log_alpha * (-log_pi_t - self.entropy_target).detach()).pow(2).mean()
            print(f'Alpha Loss: {alpha_loss}')
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

            # Count another update
            self.update_step += 1

    def update_target_function(self, q_net, target_net):
        """ Modified from SAC RLKit Implementation """
        for target_param, param in zip(target_net.parameters(), q_net.parameters()):
            target_param.data.copy_(
                target_param * (1.0 - self.tau) + param * self.tau
            )

    def sample_replay_pool(self):
        # If the agent doesn't have enough experience yet, return nothing.
        if len(self.replay_pool) < (2 * self.batch_size):
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
        torch.save(self.vae_net.state_dict(), self.save_path + 'sac_vae.net')
        torch.save(self.log_alpha, self.save_path + 'log_alpha.pt')

    def init_with_saved_weights(self):
        self.policy_network.load_state_dict(torch.load(self.save_path + 'sac_policy.net'))
        self.q_network_1.load_state_dict(torch.load(self.save_path + 'sac_q1.net'))
        self.q_network_2.load_state_dict(torch.load(self.save_path + 'sac_q2.net'))
        self.target_net_1.load_state_dict(torch.load(self.save_path + 'sac_target1.net'))
        self.target_net_2.load_state_dict(torch.load(self.save_path + 'sac_target2.net'))
        self.vae_net.load_state_dict(torch.load(self.save_path + 'sac_vae.net'))
        self.log_alpha = torch.load(self.save_path + 'log_alpha.pt')

        self.policy_network.eval()
        self.q_network_1.eval()
        self.q_network_2.eval()
        self.target_net_1.eval()
        self.target_net_2.eval()
        self.vae_net.eval()
        self.alpha = self.log_alpha.exp()

class PolicyNetwork(nn.Module):

    def __init__(self, input_size = 32, eps = 1e-6, log_std_min = -20, log_std_max = 2, init_weight = 3e-3):
        super(PolicyNetwork, self).__init__()
        self.hidden_size = 32

        # Linear layers to get to output of 2x1
        self.linear1 = nn.Linear(input_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.mean_layer = nn.Linear(self.hidden_size, 1)
        self.log_layer = nn.Linear(self.hidden_size, 1)

        # Weight initialization
        self.mean_layer.weight.data.uniform_(-init_weight, init_weight)
        self.mean_layer.bias.data.uniform_(-init_weight, init_weight)
        self.log_layer.weight.data.uniform_(-init_weight, init_weight)
        self.log_layer.bias.data.uniform_(-init_weight, init_weight)

        # Hyperparameters
        self.epsilon = eps
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, feature_input):
        x1 = F.relu(self.linear1(feature_input))
        x2 = F.relu(self.linear2(x1))
        mean = self.mean_layer(x2)
        log_std = self.log_layer(x2)

        # Create distribution
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        return (mean, std)

    def batch(self, obsv):
        # Find distributions from batch of states
        mean, std = self.forward(obsv)

        # Sample actions from the distributions
        norm_dist = torch.distributions.Normal(mean, std)
        sample = norm_dist.rsample()
        actions = torch.tanh(sample)

        # Find the log of the policy function
        log_policy = norm_dist.log_prob(sample) - torch.log(1 - actions.pow(2) + self.epsilon)
        log_policy = log_policy.sum(1, keepdim=True)

        return actions, log_policy


class SoftQFunctionNetwork(nn.Module):
    def __init__(self, init_weight = 3e-3):
        super(SoftQFunctionNetwork, self).__init__()
        action_size = 1
        input_size = (120 * 160 * 1) + action_size
        hidden_size = 64
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_weight, init_weight)
        self.linear3.bias.data.uniform_(-init_weight, init_weight)

    def forward(self, state, action):
        flat_state = torch.flatten(state, start_dim = 1)
        q_input = torch.cat([flat_state, action], 1)
        q1 = F.relu(self.linear1(q_input))
        q2 = F.relu(self.linear2(q1))
        q3 = self.linear3(q2)

        return q3

