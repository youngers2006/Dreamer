import numpy as np
import gymnasium as gym

class CarRacerAdaptor(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

    def action(self, action):
        steering = action[0]
        gas = max(0, action[1])
        brake = max(0, action[2])
        return np.array([steering, gas, brake])