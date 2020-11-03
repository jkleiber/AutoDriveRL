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

# Training parameters
XTK_POS_THRESH = 0.25
MAX_RESET_EPISODES = 10

def reward_fn(action, info, crashed):
    # Car stats
    speed = info['speed']
    lane_pos = info['cte']
    hit = info['hit']
    throttle = action[1]
    max_cte = 2.0

    # If the car crashes, return punishment
    if crashed or hit != "none":
        return -(10.0 + speed * 0.25)
    # If out of bounds, return punishment
    if fabs(lane_pos) > max_cte:
        return -(10.0 + speed * 0.25)

    # If the car is driving normally, encourage driving fast in the center
    lane_reward = (1.0 - fabs(lane_pos) / max_cte)
    reward = (lane_reward**4) * speed

    return reward

def train(agent, num_episodes, time_limit):
    # Keep track of reward per episode
    rewards = []
    pos = (0.0, 0.0, 0.0)
    num_pos_resets = 0
    obsv = env.reset()

    # Run the agent for a number of episodes
    for e in range(num_episodes):
        # Reset for next episode
        old_obsv = env.reset()
        done = False
        total_reward = 0
        action = np.array([0, 0])
        t = 0

        # Convert first observation to grayscale
        old_obsv = cv2.cvtColor(old_obsv, cv2.COLOR_BGR2GRAY)

        # Set the car position based on the most recent reset
        # Doesn't work because simulator doesn't let you set position
        # obsv = env.set_position(pos[0], pos[1], pos[2])

        # Run until the car drives off course or a time limit is reached
        while done is False and t <= time_limit:
            # Let the agent determine its action given the current observation
            action, raw_action = agent.act(old_obsv, action)

            # Execute the action and receive information back about the environment
            # obsv: vehicle front camera observation
            # reward: reward function result
            # done: did the car crash?
            # info: some diagnostics about speed, center line, etc.
            new_obsv, reward, done, info = env.step(action)

            # Grayscale the new observation
            obsv = cv2.cvtColor(new_obsv, cv2.COLOR_BGR2GRAY)

            # Calculate custom reward
            reward = reward_fn(action, info, done)

            # End episode if the car has failed miserably
            if reward < 0:
                done = True

            # Update the agent with this new experience.
            agent.add_experience(old_obsv, raw_action, reward, obsv, done)

            # Save old observation
            old_obsv = obsv

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

        # Track the reward
        rewards.append(total_reward)

        # Periodically save the agent's network data
        if e % 25 == 0:
            agent.save_weights()

        # Update the agent
        agent.update()


        # Track number of position resets
        # num_pos_resets += 1

        # if num_pos_resets > MAX_RESET_EPISODES:
        #     pos = (0.0, 0.0, 0.0)
        #     num_pos_resets = 0

    # Save the agent's data
    agent.save_weights()

    # Close the environment after the number of episodes
    env.close()

    # TODO: Plot the reward function

# Main function for training
if __name__ == "__main__":
    # Choose you agent here.
    agent = SoftActorCriticAgent()

    # Train the agent
    train(agent, 500, 2000)
