import cv2
import os
import gym
import gym_donkeycar
import numpy as np
import ImgAug
import pickle

from math import fabs

from agent import Agent
from soft_actor_critic import SoftActorCriticAgent

# Training parameters
XTK_POS_THRESH = 0.25
MAX_RESET_EPISODES = 10
EVAL_INTERVAL = 5
MAX_CTE = 2.0


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


def train(agent, num_episodes, time_limit):
    # Initialize Environment
    exe_path = f"./simulator/donkey_sim.x86_64"
    port = 9091
    conf = { "exe_path" : exe_path, "port" : port, "host" : '127.0.0.1' }
    env = gym.make("donkey-generated-roads-v0", conf=conf)

    # Keep track of reward per episode
    rewards = []
    pos = (0.0, 0.0, 0.0)
    num_pos_resets = 0
    # obsv = env.reset()
    env.set_reward_fn(calc_reward)
    pickle.dump(rewards, open("soft_actor_critic/sac_reward.pkl", 'wb'))

    # Run the agent for a number of episodes
    for e in range(num_episodes):
        # Reset for next episode
        obsv = env.reset()
        done = False
        total_reward = 0
        action = np.array([0, 0])
        t = 0

        # Run until the car drives off course or a time limit is reached
        while done is False and t <= time_limit:
            # Process the observation
            # obsv = ImgAug.detectYellow(obsv)
            # obsv = cv2.resize(obsv, (80, 80))
            obsv = cv2.cvtColor(obsv, cv2.COLOR_BGR2GRAY)

            # Let the agent determine its action given the current observation
            action, raw_action = agent.act(obsv, action, eval_mode)

            # Execute the action and receive information back about the environment
            # obsv: vehicle front camera observation
            # reward: reward function result
            # done: did the car crash?
            # info: some diagnostics about speed, center line, etc.
            new_obsv, reward, done, info = env.step(action)

            # Save the image as grayscale
            save_obsv = cv2.cvtColor(new_obsv, cv2.COLOR_BGR2GRAY)

            # End episode if the car has failed miserably
            if fabs(info['cte']) > MAX_CTE:
                done = True

            # Update the agent with this new experience.
            agent.add_experience(obsv, raw_action, reward, save_obsv, done)

            # Save old observation
            obsv = new_obsv

            # Update reward
            total_reward += reward

            # Increment time
            t += 1

        if t > time_limit:
            print('Time Limit Reached!')
        else:
            print('Crashed!')

        # Print results of the episode
        print(f'Episode {e} over after {t} steps, Total Reward: {total_reward}')

        # Track the reward
        rewards.append(total_reward)

        # Save the agent's network data
        agent.save_weights()
        pickle.dump(rewards, open("soft_actor_critic/sac_reward.pkl", 'wb'))

        # Update the agent
        agent.update()

    # Save the agent's data
    agent.save_weights()

    # Save the rewards
    pickle.dump(rewards, open("soft_actor_critic/sac_reward.pkl", 'wb'))

    # Close the environment after the number of episodes
    env.close()

# Main function for training
if __name__ == "__main__":
    # Setup SAC
    agent = SoftActorCriticAgent()

    # Train the agent for 400 episodes
    train(agent, 400, 2000)
