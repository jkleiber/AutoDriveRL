import gym
import gym_donkeycar
import examples.reinforcement_learning.sac_train as sac_trainer

# Create Gym environment
exe_path = f"./simulator/donkey_sim.x86_64"
weightPath = None #'obddqnAN3.h5'
runTest = False
port = 9091
conf = { "exe_path" : exe_path, "port" : port, "host" : '127.0.0.1' }
throttle = .05
envName = "donkey-generated-roads-v0"

if __name__ == '__main__':

    args = [exe_path,weightPath,runTest,port,throttle,envName]
    sac_trainer.run_ppo(args)
