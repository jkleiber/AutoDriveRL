
import numpy as np

class Agent:

    def __init__(self):
        pass

    def act(self, obsv):
        # Drive straight with small speed
        return np.array([0.0,0.9])