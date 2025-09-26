class MujocoAdapter:
    def __init__(self, custom_env):
        self.env = custom_env

    def reset(self):
        print("Adapter: Calling reset(), translating to new_game().")
        obs = self.env.new_game()
        return obs, {}
    
    def step(self, action):
        print("Adapter: Calling step(), translating to advance_frame().")
        result = self.env.advance_frame(action)
        obs = result["state"]
        reward = result["reward"]
        terminated = result["done"]
        truncated = False 
        info = {}
        return obs, reward, terminated, truncated, info