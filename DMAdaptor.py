import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dm_control import suite
from dm_env import specs, TimeStep
from collections import OrderedDict

class DMControlAdapter(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        dm_env,
        render_camera_id: int = 0,
        render_height: int = 240,
        render_width: int = 320,
    ):
        if suite is None:
            raise ImportError("dm_control is not installed. Please follow installation instructions.")

        self._env = dm_env
        
        self.observation_spec = self._env.observation_spec()
        self.observation_space = self._spec_to_box(list(self.observation_spec.values()))

        self.action_spec = self._env.action_spec()
        self.action_space = spaces.Box(
            low=self.action_spec.minimum,
            high=self.action_spec.maximum,
            shape=self.action_spec.shape,
            dtype=np.float32,
        )
        
        # Rendering settings
        self.render_mode = "rgb_array"
        self._render_camera_id = render_camera_id
        self._render_height = render_height
        self._render_width = render_width

    @staticmethod
    def _spec_to_box(spec_list, dtype=np.float32):
        total_dim = sum(int(np.prod(s.shape)) for s in spec_list)
        low = np.full(total_dim, -np.inf, dtype=dtype)
        high = np.full(total_dim, np.inf, dtype=dtype)
        
        current_dim = 0
        for s in spec_list:
            if isinstance(s, specs.BoundedArray):
                dim = int(np.prod(s.shape))
                low[current_dim : current_dim + dim] = np.broadcast_to(s.minimum, dim)
                high[current_dim : current_dim + dim] = np.broadcast_to(s.maximum, dim)
                current_dim += dim
    
        return spaces.Box(low=low, high=high, shape=(total_dim,), dtype=dtype)

    @staticmethod
    def _flatten_obs(obs_dict: OrderedDict):
        obs_parts = []
        for obs in obs_dict.values():
            flat_obs = obs.ravel()
            obs_parts.append(flat_obs)
        return np.concatenate(obs_parts, axis=0, dtype=np.float32)

    def step(self, action):
        action = np.clip(action, self.action_spec.minimum, self.action_spec.maximum)
        
        time_step = self._env.step(action)
        
        observation = self._flatten_obs(time_step.observation)
        reward = time_step.reward or 0.0
    
        terminated = False
        truncated = time_step.last()
        
        info = {}

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        time_step = self._env.reset()
        observation = self._flatten_obs(time_step.observation)
        info = {}
        
        return observation, info

    def render(self):
        if self.render_mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {self.render_mode}. Only 'rgb_array' is available.")
            
        return self._env.physics.render(
            height=self._render_height,
            width=self._render_width,
            camera_id=self._render_camera_id,
        )

    def close(self):
        self._env.close()