# AutoDriveRL

Autonomous driving using Reinforcement Learning and DonkeyCar.

## Get Started

**Prerequisites**  
This code must be run on Linux

**Installation Steps**

1. Install [Pipenv](https://pypi.org/project/pipenv/)
2. Run `cd src/`
3. Install the simulator and third party dependencies with `./setup.sh`
4. Test the simulator by running `pipenv run python test_sim.py`

If the simulator opens, then you can close it. You're ready to train!

## Algorithm Training and Evaluation

There are 3 algorithms available for training:
* Soft Actor Critic
* DDQN
* PPO

To run any algorithm, first make sure you are inside the `src/` directory. Pipenv needs to reference the `Pipfile` in that folder to manage its virtual environment.

To train Soft Actor Critic, simply run
```
pipenv run python sac_custom.py
```
To demonstrate the results you get, you can run `sac_demo.py`.

There is also an implementation of Soft Actor Critic based on stable-baselines3. We found that this worked less well than our custom implementation with the VAE. To run this you can command
```
pipenv run python sac_sb3.py
```
and to evaluate it you have to set `runTest = True` in `sac_sb3.py`.

For DDQN:
```
pipenv run python obddqn.py
```
DDQN runs from pre-trained weights, so this defaults to evaluation mode. To train from scratch, you can set `runInferenceDontTrain = False` in `obddqn.py`.

and for PPO:
```
pipenv run python ppo_sb3.py
```
This is based on stable-baselines3. To evaluate this algorithm you can set `runTest = True` in `ppo_sb3.py`.

## Common Problems
Sometimes the simulator will open and then close when you are training / evaluating. Just retry the command you are running and it will work eventually. This is caused by some TCP handshaking the simulator and the gym have to do and sometimes the operating system will block it for too long and the simulator will give up.

## Pre-trained Networks
We provide pre-trained networks for DDQN, PPO, and SAC (stable-baselines3 version) in the `src` folder. The custom SAC pretrained network data is provided in `src/soft-actor-critic`, which is where it is pulled from and saved to by `sac_custom.py` and `sac_demo.py`.

## Acknowledgements

The DonkeyCar Gym is available from [Tawn Kramer's gym-donkeycar repository](https://github.com/tawnkramer/gym-donkeycar) and had some starter code for DDQN and PPO.    

[stable-baselines3](https://github.com/DLR-RM/stable-baselines3) helped us evaluate our implementations against baseline algorithms

[VAE-CNN](https://github.com/sksq96/pytorch-vae/blob/master/vae-cnn.ipynb) was a project that helped with some implementation details of the VAE.

[PolicyGradientMethods](https://github.com/cyoon1729/Policy-Gradient-Methods) by Chris Yoon was helpful in some implementation details for Soft Actor Critic.

## Contact us

If you want to contact the developers we can be reached at
* Justin Kleiber - [jkleiber@vt.edu](mailto:jkleiber@vt.edu)
* Sami Wood - [samiw@vt.edu](mailto:samiw@vt.edu)