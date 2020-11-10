# AutoDriveRL

Autonomous driving using Reinforcement Learning and DonkeyCar.

## Get Started

1. Install Pipenv
2. Run `cd src/`
3. Install the simulator and third party dependencies with `./setup.sh`
4. Test the simulator by running `pipenv run python test_sim.py`

## Training

There is 1 algorithm available for training:
* Soft Actor Critic

To train, simply run
```
pipenv run python train.py
```

## Evaluation

To demonstrate a trained agent, run
```
pipenv run python demo.py
```

## Acknowledgements

The DonkeyCar Gym is available from [Tawn Kramer's gym-donkeycar repository](https://github.com/tawnkramer/gym-donkeycar)