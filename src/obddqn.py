import gym
import gym_donkeycar
import reinforcement_learning.ddqn as dq


from PolicyGradientMethods.common.utils import mini_batch_train  # import training function
from PolicyGradientMethods.td3.td3 import TD3Agent  # import agent from algorithm of interest

# Create Gym environment
exe_path = f"./simulator/donkey_sim.x86_64"
weightPath = 'obddqn.h5'
runInferenceDontTrain = False
port = 9091
conf = { "exe_path" : exe_path, "port" : port, "host" : '127.0.0.1' }
throttle = .2
envName = "donkey-generated-track-v0"
#env = gym.make(envName, conf=conf)

if __name__ == '__main__':

    args = [exe_path,weightPath,runInferenceDontTrain,port,throttle,envName]
    dq.run_ddqn(args)

