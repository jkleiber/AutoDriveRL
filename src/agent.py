
import numpy as np

class Agent:

    def __init__(self):
        pass

    def act(self, obsv):
        # Drive straight with small speed
        return np.array([0.0,0.9])

    def update(self, old_obsv, action, reward, new_obsv, is_crashed):
        pass

    def save_weights(self):
        pass

    def init_with_saved_weights(self):
        pass