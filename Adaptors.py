import numpy as np
import gymnasium as gym

class CarRacerAdaptor(gym.ActionWrapper):
    def __init__(self, env):
        super.__init__(env)

    def action(self, action):
        steering = action[0]
        gas = (action[1] + 1) / 2
        brake = (action[2] + 1) / 2
        return np.array([steering, gas, brake])