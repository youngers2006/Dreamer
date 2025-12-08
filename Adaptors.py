import numpy as np
import gymnasium as gym

class CarRacerAdaptor(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

    def action(self, action):
        steering = action[0]
        gas = (action[1] + 1) / 2
        brake = (action[2] + 1) / 2
        return np.array([steering, gas, brake])
    
class ActionRepeat(gym.Wrapper):
    def __init__(self, env, repeat=4):
        super().__init__(env)
        self.repeat = repeat

    def step(self, action):
        total_reward = 0.0
        done = False
        truncated = False
        last_obs = None
        last_info = {}
        
        for _ in range(self.repeat):
            obs, reward, d, t, info = self.env.step(action)
            total_reward += reward
            done = done or d
            truncated = truncated or t
            last_obs = obs
            last_info = info
            if done or truncated:
                break
        
        return last_obs, total_reward, done, truncated, last_info