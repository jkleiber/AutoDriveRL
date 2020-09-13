import os
import gym
import gym_donkeycar
import numpy as np

#%% SET UP ENVIRONMENT
exe_path = f"./simulator/donkey_sim.x86_64"
port = 9091

conf = { "exe_path" : exe_path, "port" : port, "host" : '127.0.0.1' }

env = gym.make("donkey-generated-track-v0", conf=conf)

#%% PLAY
obv = env.reset()
for t in range(100):
    action = np.array([0.0,0.5]) # drive straight with small speed
# execute the action
obv, reward, done, info = env.step(action)