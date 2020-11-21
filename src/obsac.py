import gym
import gym_donkeycar

from PolicyGradientMethods.common.utils import mini_batch_train  # import training function
from PolicyGradientMethods.sac.sac2020 import SACAgent

import pickle
import ImgAug
import torch
import cv2
import numpy as np

# Create Gym environment
exe_path = f"./simulator/donkey_sim.x86_64"
port = 9091
conf = { "exe_path" : exe_path, "port" : port, "host" : '127.0.0.1' }
env = gym.make("donkey-generated-roads-v0", conf=conf)

last = [0,0,0,0]
def calc_reward(self, d):
    if d:
        return -1.0

    if self.cte > self.max_cte:
        return -1.0

    if self.hit != "none":
        return -2.0

    # going fast close to the center of lane yeilds best reward
    if self.cte== 0:
        return 0
    #print(last)
    delCte = self.cte - last[0]
    delAbsCte = abs(last[0]) - abs(self.cte)
    distDelta = np.linalg.norm(np.array([self.x,self.y,self.z])-np.array([last[1],last[2],last[3]]))
    headErr = np.rad2deg(np.arcsin(delCte/distDelta))

    if headErr == 0:
        headErr = .001
    if np.isnan(headErr):
        headErr = 1000
    #print('heading err' + str(headErr))
    last[0] = self.cte
    last[1] = self.x
    last[2] = self.y
    last[3] = self.z

    # if headErr > 45:
    #     return -1000
    #return delAbsCte
    return 10-abs(headErr) +min(100,1/abs(max(.01,.1*self.cte**2))) #+ 10/abs(headErr) #derivitive of cte like heading error



def mini_batch_train_car(env, agent, max_episodes, max_steps, batch_size):
    episode_rewards = []

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Pre-process car image
            state = ImgAug.detectYellow(state)
            state = cv2.resize(state, (32, 32))
            # save_state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
            state = state.flatten()
            print(state.shape)
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)

            # Process the saved state
            save_state = ImgAug.detectYellow(next_state)
            save_state = cv2.resize(save_state, (32, 32))
            # save_state = cv2.cvtColor(save_state, cv2.COLOR_BGR2GRAY)
            save_state = save_state.flatten()

            # Save to replay buffer
            agent.replay_buffer.push(state, action, reward, next_state, done)
            episode_reward += reward

            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)   

            if done or step == max_steps-1:
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward))
                break

            if step % 20 == 0:
                agent.save_weights()

            state = next_state

    return episode_rewards

if __name__ == '__main__':

    # Initialize agent
    gamma = 0.99
    tau = 1e-2
    alpha = 0.5
    buffer_maxlen = 100000
    q_lr = 1e-3
    policy_lr = 1e-3
    a_lr = 1e-3

    agent = SACAgent(env, gamma, tau, alpha, q_lr, policy_lr, a_lr, buffer_maxlen)

    # Define training parameters
    max_episodes = 100
    max_steps = 500
    batch_size = 32

    # Set the custom reward
    env.set_reward_fn(calc_reward)

    # Train agent with mini_batch_train function
    episode_rewards = mini_batch_train_car(env, agent, max_episodes, max_steps, batch_size)

    # Save everything
    agent.save_weights()
    pickle.dump(episode_rewards, open('soft_actor_critic/ob_sac_rewards.pkl', 'wb'))
