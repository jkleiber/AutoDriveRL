'''
file: sac_train.py
author: Justin Kleiber
date: 25 November 2020
notes: sac test from stable-baselines3
'''
import os
import argparse
import gym
import gym_donkeycar
import uuid
import ImgAug
import cv2
import numpy as np

from stable_baselines3 import PPO, SAC

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


def run_sac(args):

	# Initialize the donkey environment
    # where env_name one of:    
    env_list = [
       "donkey-warehouse-v0",
       "donkey-generated-roads-v0",
       "donkey-avc-sparkfun-v0",
       "donkey-generated-track-v0",
       "donkey-mountain-track-v0"
    ]

    argsSimIdx = 0
    argsWeightPathIdx =1
    argsTestBoolIdx = 2
    argsportIdx = 3
    argsThrottleIdx = 4
    argsEnvNameIdx = 5
    
    env_id = args[argsEnvNameIdx]

    conf = {"exe_path" : args[argsSimIdx],
        "host" : "127.0.0.1",
        "port" : args[argsportIdx],

        "body_style" : "donkey",
        "body_rgb" : (128, 128, 128),
        "car_name" : "me",
        "font_size" : 100,

        "racer_name" : "SAC",
        "country" : "USA",
        "bio" : "Learning to drive w SAC RL",
        "guid" : str(uuid.uuid4()),

        "max_cte" : 5,
        }

    # TODO: This is for testing, but not implemented yet
    if args[argsTestBoolIdx]:
        print('Entering Test mode')
        #Make an environment test our trained policy
        env = gym.make(args[argsEnvNameIdx], conf=conf)
        model = SAC.load("sac_donkey")
    
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            env.render()

        print("done testing")
        
    else:
    
        #make gym env
        env = gym.make(args[argsEnvNameIdx], conf=conf)

        # Set reward function
        env.set_reward_fn(calc_reward)

        #create cnn policy
        model = SAC('CnnPolicy', env, verbose=1, buffer_size=30000)

        #set up model in learning mode with goal number of timesteps to complete
        model.learn(total_timesteps=10000)
        print('training over')
        model.save("sac_donkey")

        # obs = env.reset()
        
        # num_episodes = 10000
        # for e in range(num_episodes):
        #     while not done:
        #         action, _states = model.predict(obs)
                
        #         obs, rewards, done, info = env.step(action) 
                    
        #     # Save the current model
        #     model.save("sac_donkey")

        #     if e % 10 == 0:
        #         # give random new env (prevent overfitting)
        #         env.unwrapped.close()
        #         env = gym.make(args[argsEnvNameIdx], conf=conf)
        #         env.set_reward_fn(calc_reward)

        # # Save the agent
        # model.save("sac_donkey")
        # print("done training")


    env.close()
