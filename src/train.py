import cv2
import os
import gym
import gym_donkeycar
import numpy as np

from math import fabs

from agent import Agent
from soft_actor_critic import SoftActorCriticAgent

# Initialize Environment
exe_path = f"./simulator/donkey_sim.x86_64"
port = 9091
conf = { "exe_path" : exe_path, "port" : port, "host" : '127.0.0.1' }
env = gym.make("donkey-generated-roads-v0", conf=conf)

def reward_fn(action, info, crashed):
    # Car stats
    speed = info['speed']
    lane_pos = info['cte']
    hit = info['hit']
    throttle = action[1]
    max_cte = 5.0

    # If the car crashes, return punishment
    if crashed or hit != "none":
        return -1.0
    # If out of bounds, return punishment
    if fabs(lane_pos) > max_cte:
        return -1.0
    # Don't reward backward driving
    if throttle < 0.0:
        return -0.5

    # If the car is driving normally, encourage driving fast in the center
    lane_reward = (1.0 - fabs(lane_pos) / max_cte)
    reward = lane_reward * speed * 0.1
    # reward = speed * (0.1 / (fabs(1.0 - lane_reward) + 0.1))

    return reward

def train(agent, num_episodes, time_limit):
    # Run the agent for a number of episodes
    for e in range(num_episodes):
        # Reset environment and simulation
        obsv = env.reset()
        done = False
        total_reward = 0
        t = 0

        # Run until the car drives off course or a time limit is reached
        while done is False and t <= time_limit:
            # Let the agent determine its action given the current observation
            action = agent.act(obsv)
            old_obsv = obsv

            # Execute the action and receive information back about the environment
            # obsv: vehicle front camera observation
            # reward: reward function result
            # done: did the car crash?
            # info: some diagnostics about speed, center line, etc.
            obsv, reward, done, info = env.step(action)

            # Calculate custom reward
            reward = reward_fn(action, info, done)

            # Update the agent with this new experience.
            agent.add_experience(old_obsv, action, reward, obsv, done)

            # Show what the agent sees.
            # cv2.imshow('DonkeyCar Camera', obsv)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

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

        # Update the agent
        agent.update()

    # Save the agent's data
    agent.save_weights()

    # Close the environment after the number of episodes
    env.close()

# Main function for training
if __name__ == "__main__":
    # Choose you agent here.
    agent = SoftActorCriticAgent()

    # Train the agent
    train(agent, 100, 200)
