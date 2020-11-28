
import cv2
import os
import gym
import gym_donkeycar
import numpy as np

from agent import Agent
from soft_actor_critic import SoftActorCriticAgent

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


def demo(agent, time_limit):
    # Initialize Environment
    exe_path = f"./simulator/donkey_sim.x86_64"
    port = 9091
    conf = { "exe_path" : exe_path, "port" : port, "host" : '127.0.0.1' }
    env = gym.make("donkey-generated-roads-v0", conf=conf)

    # Run the agent once
    # Reset environment and simulation
    obsv = env.reset()
    done = False
    total_reward = 0
    action = np.array([0, 0])
    t = 0

    env.set_reward_fn(calc_reward)

    # Run until the car drives off course or a time limit is reached
    while done is False and t <= time_limit:
        obsv = cv2.cvtColor(obsv, cv2.COLOR_BGR2GRAY)

        # Let the agent determine its action given the current observation
        action, _ = agent.act(obsv, action, False)

        obsv, reward, done, info = env.step(action)

        # Update reward
        total_reward += reward

        # Increment time
        t += 1

    print(f'Demonstration ended after {t} steps, Total Reward: {total_reward}')

    # Close the environment after the number of episodes
    env.close()

# Main function for training
if __name__ == "__main__":
    # Choose you agent here.
    agent = SoftActorCriticAgent()

    # Load from saved weights
    agent.init_with_saved_weights()

    # Show off the agent's abilities
    demo(agent, 1000)
