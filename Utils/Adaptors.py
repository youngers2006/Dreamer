import numpy as np
import gymnasium as gym
import cv2
import PyFlyt.gym_envs

class DroneAdaptor(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

    def action(self, action):
        forward = action[0]  # Forward/Back
        right = action[1]  # Strafe Right/Left (New!)
        lift = action[2]  # Up/Down
        turn = action[3]  # Yaw Rotation
        
        vel_x = forward
        vel_y = right 
        vel_z = lift
        yaw_rate = turn

        return np.array([vel_x, vel_y, vel_z, yaw_rate], dtype=np.float32)

class CarRacerAdaptor(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

    def action(self, action):
        steering = action[0]
        gas = (action[1] + 1) / 2
        brake = (action[2] + 1) / 2
        return np.array([steering, gas, brake])
    
class CropObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # CarRacing-v2 defaults to 96x96.
        # We crop the bottom 12 pixels (dashboard), leaving 84x96.
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 96, 3), dtype=np.uint8
        )

    def observation(self, obs):
        # Keep only the top 84 rows
        return obs[:84, :, :]
    
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