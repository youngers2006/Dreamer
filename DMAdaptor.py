import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dm_control import suite
from dm_env import specs, TimeStep
from collections import OrderedDict

class DMControlAdapter(gym.Env):
    """
    An adapter class that wraps an existing DeepMind Control Suite environment to
    make it compatible with the Gymnasium API.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        dm_env,
        render_camera_id: int = 0,
        render_height: int = 240,
        render_width: int = 320,
    ):
        """
        Initializes the adapter.

        Args:
            dm_env: An existing dm_control environment instance.
            render_camera_id (int): The ID of the camera to use for rendering.
            render_height (int): The height of the rendered image.
            render_width (int): The width of the rendered image.
        """
        if suite is None:
            raise ImportError("dm_control is not installed. Please follow installation instructions.")

        self._env = dm_env
        
        # Define observation and action spaces from the existing environment
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
        """
        Converts a list of dm_control specs to a single gymnasium.spaces.Box.
        Flattens the observations into a single vector.
        """
        total_dim = sum(int(np.prod(s.shape)) for s in spec_list)
        low = np.full(total_dim, -np.inf, dtype=dtype)
        high = np.full(total_dim, np.inf, dtype=dtype)
        
        # Set bounds for bounded specs
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
        """
        Flattens an OrderedDict of observations into a single numpy array.
        """
        obs_parts = []
        for obs in obs_dict.values():
            flat_obs = obs.ravel()
            obs_parts.append(flat_obs)
        return np.concatenate(obs_parts, axis=0, dtype=np.float32)

    def step(self, action):
        """
        Takes a step in the environment.

        Returns:
            A 5-tuple (observation, reward, terminated, truncated, info).
        """
        # dm_control expects actions to be within the spec bounds
        action = np.clip(action, self.action_spec.minimum, self.action_spec.maximum)
        
        time_step = self._env.step(action)
        
        observation = self._flatten_obs(time_step.observation)
        reward = time_step.reward or 0.0
        
        # In dm_control, an episode ends due to a time limit (truncation).
        # There is no concept of a terminal state due to task completion.
        terminated = False
        truncated = time_step.last()
        
        info = {}

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Resets the environment.

        Returns:
            A 2-tuple (observation, info).
        """
        # Note: dm_control environments are seeded at initialization.
        # The `seed` argument here is for API compatibility but doesn't re-seed.
        super().reset(seed=seed)
        
        time_step = self._env.reset()
        observation = self._flatten_obs(time_step.observation)
        info = {}
        
        return observation, info

    def render(self):
        """
        Renders the environment.

        Returns:
            An RGB array of the current environment state.
        """
        if self.render_mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {self.render_mode}. Only 'rgb_array' is available.")
            
        return self._env.physics.render(
            height=self._render_height,
            width=self._render_width,
            camera_id=self._render_camera_id,
        )

    def close(self):
        """
        Closes the environment and releases resources.
        """
        self._env.close()


# --- Example Usage ---
def main():
    """
    Demonstrates how to use the DMControlAdapter.
    """
    if suite is None:
        print("Cannot run example because dm_control is not installed.")
        return

    print("Creating dm_control 'cartpole-balance' environment...")
    # 1. Create the original dm_control environment
    dm_env = suite.load(domain_name="cartpole", task_name="balance")

    print("Wrapping the environment with the adapter...")
    # 2. Wrap it with the adapter
    env = DMControlAdapter(dm_env)

    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")

    # Standard Gymnasium loop
    obs, info = env.reset()
    total_reward = 0
    
    for i in range(200):
        # Sample a random action
        action = env.action_space.sample()
        
        # Take a step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Simple check to see if we can render
        if i == 0:
            img = env.render()
            print(f"Rendered image shape: {img.shape}")
            # To display the image, you would use a library like matplotlib or Pillow
            # from PIL import Image
            # Image.fromarray(img).show()

        if terminated or truncated:
            print(f"Episode finished after {i+1} steps. Total reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
    
    env.close()
    print("Environment closed.")