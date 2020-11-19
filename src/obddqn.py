import gym
import gym_donkeycar
import reinforcement_learning.ddqn as dq


from PolicyGradientMethods.common.utils import mini_batch_train  # import training function
from PolicyGradientMethods.td3.td3 import TD3Agent  # import agent from algorithm of interest

# Create Gym environment
exe_path = f"./simulator/donkey_sim.x86_64"
weightPath = 'obddqnAN3.h5'
runInferenceDontTrain = True
port = 9091
conf = { "exe_path" : exe_path, "port" : port, "host" : '127.0.0.1' }
throttle = .05
envName = "donkey-generated-roads-v0"
#env = gym.make(envName, conf=conf)

if __name__ == '__main__':

    args = [exe_path,weightPath,runInferenceDontTrain,port,throttle,envName]
    dq.run_ddqn(args)

