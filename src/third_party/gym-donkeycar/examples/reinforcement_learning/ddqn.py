'''
file: ddqn.py
author: Felix Yu
date: 2018-09-12
original: https://github.com/flyyufelix/donkey_rl/blob/master/donkey_rl/src/ddqn.py
'''
import os
import sys
import random
import argparse
import signal
import uuid
import math

import numpy as np
import gym
import cv2
import ImgAug

from collections import deque
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D
import tensorflow as tf
from tensorflow.keras import backend as K

import gym_donkeycar

EPISODES = 10000
img_rows, img_cols = 32, 32
#extract only yellow and convert to balck and white small array with simple features to improve performace
# Convert image into Black and white
img_channels = 16 # stack 16 frames ~.5 seconds of video

class DQNAgent:

    def __init__(self, state_size, action_space, train=True):
        self.t = 0
        self.max_Q = 0
        self.train = train

        # Get size of state and action
        self.state_size = state_size
        self.action_space = action_space
        self.action_size = action_space

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 1e-4
        if (self.train):
            self.epsilon = 1.0
            self.initial_epsilon = 1.0
        else:
            self.epsilon = 1e-6
            self.initial_epsilon = 1e-6
        self.epsilon_min = 0.02
        self.batch_size = 32 # number of frames to preform grad desent on
        self.train_start = 100 # number of frames before training starts
        self.explore = 8000 # rate at which actions transition from random to determined

        # Create replay memory using deque
        self.memory = deque(maxlen=5000)

        # Create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # Copy the model to target model
        # --> initialize the target model so that the parameters of model & target model to be same
        self.update_target_model()


    def build_model(self):
        # Use CNN to map 32,32 image into 15 discreate actions
        model = Sequential()
        model.add(Conv2D(24, (5, 5), strides=(2, 2), padding="same",input_shape=(img_rows,img_cols,img_channels)))  #80*80*4
        model.add(Activation('relu'))
        model.add(Conv2D(32, (5, 5), strides=(2, 2), padding="same"))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (5, 5), strides=(2, 2), padding="same"))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same"))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))

        # 15 categorical bins for Steering angles
        model.add(Dense(15, activation="linear"))

        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse',optimizer=adam)

        return model


    def rgb2gray(self, rgb):
        '''
        take a numpy rgb image return a new single channel image converted to greyscale
        '''
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


    def process_image(self, obs):
        #obs = self.rgb2gray(obs)
        obs = cv2.resize(obs, (img_rows, img_cols))
        return obs


    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # Get action from model using epsilon-greedy policy
    def get_action(self, s_t):
        if np.random.rand() <= self.epsilon:
            print('random')
            return self.action_space.sample()[0]
        else:
            #print("Return Max Q Prediction")
            q_value = self.model.predict(s_t)

            # Convert q array to steering value
            return linear_unbin(q_value[0])


    def replay_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= (self.initial_epsilon - self.epsilon_min) / self.explore


    def train_replay(self):
        if len(self.memory) < self.train_start:
            return

        batch_size = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)

        state_t, action_t, reward_t, state_t1, terminal = zip(*minibatch)
        state_t = np.concatenate(state_t)
        state_t1 = np.concatenate(state_t1)
        targets = self.model.predict(state_t)
        self.max_Q = np.max(targets[0])
        target_val = self.model.predict(state_t1)
        target_val_ = self.target_model.predict(state_t1)
        for i in range(batch_size):
            if terminal[i]:
                targets[i][action_t[i]] = reward_t[i]
            else:
                a = np.argmax(target_val[i])
                targets[i][action_t[i]] = reward_t[i] + self.discount_factor * (target_val_[i][a])

        self.model.train_on_batch(state_t, targets)


    def load_model(self, name):
        self.model.load_weights(name)


    # Save the model which is under training
    def save_model(self, name):
        self.model.save_weights(name)



## Utils Functions ##

def linear_bin(a):
    """
    Convert a value to a categorical array.

    Parameters
    ----------
    a : int or float
        A value between -1 and 1

    Returns
    -------
    list of int
        A list of length 15 with one item set to 1, which represents the linear value, and all other items set to 0.
    """
    a = a + 1
    b = round(a / (2 / 14))
    arr = np.zeros(15)
    arr[int(b)] = 1
    return arr


def linear_unbin(arr):
    """
    Convert a categorical array to value.

    See Also
    --------
    linear_bin
    """
    if not len(arr) == 15:
        raise ValueError('Illegal array length, must be 15')
    b = np.argmax(arr)
    a = b * (2 / 14) - 1
    return a


def run_ddqn(args):
    '''
    run a DDQN training session, or test it's result, with the donkey simulator
    '''

    # only needed if TF==1.13.1
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config));
    # sess = tf.compat.v1.Session(config=config)
    # K.set_session(sess)
    argsSimIdx = 0
    argsWeightPathIdx =1
    argsTestBoolIdx = 2
    argsportIdx = 3
    argsThrottleIdx = 4
    argsEnvNameIdx = 5

    conf = {"exe_path" : args[argsSimIdx],
        "host" : "127.0.0.1",
        "port" : args[argsportIdx],

        "body_style" : "donkey",
        "body_rgb" : (128, 128, 128),
        "car_name" : "me",
        "font_size" : 100,

        "racer_name" : "DDQN",
        "country" : "USA",
        "bio" : "Learning to drive w DDQN RL",
        "guid" : str(uuid.uuid4()),

        "max_cte" : 10,
        }


    # Construct gym environment. Starts the simulator if path is given.
    env = gym.make(args[argsEnvNameIdx], conf=conf)
    last = [0,0,0,0]
    def calc_reward(self, d):
        # define a new reward function
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
        headErr = np.rad2deg(np.arcsin(delCte/distDelta)) # calculate the heading error of the cart

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
        # calculate reward function that accounts for heading and cross track errors
        return 10-abs(headErr) +min(100,1/abs(max(.01,.1*self.cte**2))) #+ 10/abs(headErr) #derivitive of cte like heading error

    env.set_reward_fn(calc_reward)

    # not working on windows...
    def signal_handler(signal, frame):
        print("catching ctrl+c")
        env.unwrapped.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGABRT, signal_handler)

    # Get size of state and action from environment
    state_size = (img_rows, img_cols, img_channels)
    action_space = env.action_space # Steering and Throttle

    try:
        agent = DQNAgent(state_size, action_space, train=not args[argsTestBoolIdx])

        throttle = args[argsThrottleIdx] # Set throttle as constant value

        episodes = []

        if os.path.exists(args[argsWeightPathIdx]):
            print("load the saved model")
            agent.load_model(args[argsWeightPathIdx])

        for e in range(EPISODES):

            print("Episode: ", e)

            done = False
            obs = env.reset()

            episode_len = 0

            #obs = ImgAug.preProcessRGB(obs)
            obs = ImgAug.detectYellow(obs)#given simulated camera frame detect center line
            x_t = agent.process_image(obs)#resize to 32x32

            s_t = np.stack([x_t for x in range(img_channels)],axis=2) # stack last n frames
            # In Keras, need to reshape
            s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2]) #1*80*80*4

            while not done:

                # Get action for the current state and go one step in environment
                steering = agent.get_action(s_t)
                action = [steering, throttle]
                # print(action)
                next_obs, reward, done, info = env.step(action)
                print(reward)

                #next_obs = ImgAug.preProcessRGB(next_obs)
                next_obs = ImgAug.detectYellow(next_obs)
                x_t1 = agent.process_image(next_obs)

                x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1) #1x80x80x1
                s_t1 = np.append(x_t1, s_t[:, :, :, :img_channels-1], axis=3) #1x80x80x4

                # Save the sample <s, a, r, s'> to the replay memory
                agent.replay_memory(s_t, np.argmax(linear_bin(steering)), reward, s_t1, done)
                agent.update_epsilon()

                if agent.train:
                    agent.train_replay()

                s_t = s_t1
                agent.t = agent.t + 1
                episode_len = episode_len + 1
                if agent.t % 30 == 0:
                    print("EPISODE",  e, "TIMESTEP", agent.t,"/ ACTION", action, "/ REWARD", reward, "/ EPISODE LENGTH", episode_len, "/ Q_MAX " , agent.max_Q)

                if done:

                    # Every episode update the target model to be same with model
                    agent.update_target_model()

                    episodes.append(e)


                    # Save model for each episode
                    if agent.train:
                        agent.save_model(args[argsWeightPathIdx])

                    print("episode:", e, "  memory length:", len(agent.memory),
                        "  epsilon:", agent.epsilon, " episode length:", episode_len)

                    if e % 10 == 0:
                        # give random new env [stop overfit I think]
                        env.unwrapped.close()
                        env = gym.make(args[argsEnvNameIdx], conf=conf)
                        env.set_reward_fn(calc_reward)


    except KeyboardInterrupt:
        print("stopping run...")
    finally:
        env.unwrapped.close()


if __name__ == "__main__":

    # Initialize the donkey environment
    # where env_name one of:
    env_list = [
       "donkey-warehouse-v0",
       "donkey-generated-roads-v0",
       "donkey-avc-sparkfun-v0",
       "donkey-generated-track-v0"
       "donkey-mountain-track-v0"
    ]

    parser = argparse.ArgumentParser(description='ddqn')
    parser.add_argument('--sim', type=str, default="manual", help='path to unity simulator. maybe be left at manual if you would like to start the sim on your own.')
    parser.add_argument('--model', type=str, default="rl_driver.h5", help='path to model')
    parser.add_argument('--test', action="store_true", help='agent uses learned model to navigate env')
    parser.add_argument('--port', type=int, default=9091, help='port to use for websockets')
    parser.add_argument('--throttle', type=float, default=0.3, help='constant throttle for driving')
    parser.add_argument('--env_name', type=str, default='donkey-mountain-track-v0', help='name of donkey sim environment', choices=env_list)

    args = parser.parse_args()

    run_ddqn(args)



