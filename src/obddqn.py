import gym
import gym_donkeycar
import examples.reinforcement_learning.ddqn as dq


# Create Gym environment
exe_path = f"./simulator/donkey_sim.x86_64"
weightPath = 'obddqnAN3.h5'
runInferenceDontTrain = True
port = 9091
conf = { "exe_path" : exe_path, "port" : port, "host" : '127.0.0.1' }
#set a constant value for throttle and only worry about finding the sterrring angles
throttle = .05
envName = "donkey-generated-roads-v0"
#env = gym.make(envName, conf=conf)

if __name__ == '__main__':
    #make a call to the simulator
    args = [exe_path,weightPath,runInferenceDontTrain,port,throttle,envName]
    dq.run_ddqn(args)

