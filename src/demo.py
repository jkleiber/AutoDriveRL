
import os
import gym
import gym_donkeycar
import numpy as np

from agent import Agent
from soft_actor_critic import SoftActorCriticAgent

# Initialize Environment
exe_path = f"./simulator/donkey_sim.x86_64"
port = 9091
conf = { "exe_path" : exe_path, "port" : port, "host" : '127.0.0.1' }
env = gym.make("donkey-generated-track-v0", conf=conf)


def demo(agent, time_limit):
    # Run the agent once
    # Reset environment and simulation
    obsv = env.reset()
    done = False
    total_reward = 0
    t = 0

    # Run until the car drives off course or a time limit is reached
    while done is False and t <= time_limit:
        # Let the agent determine its action given the current observation
        action = agent.act(obsv)

        # Execute the action and receive information back about the environment
        # obsv: vehicle front camera observation
        # reward: reward function result
        # done: did the car crash?
        # info: some diagnostics about speed, etc.
        #
        # Note: you can customize reward function with `env.set_reward_fn()`
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
    demo(agent, 100)
