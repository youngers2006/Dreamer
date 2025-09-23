import torch
import torch.nn as nn
from WorldModel import WorldModel
from Agent import Actor, Critic

class Dreamer(nn.Module):
    def __init__(self):
        self.world_model = WorldModel()
        self.actor = Actor()
        self.critic = Critic()
    
    def imagine_episode(self):
        hidden_state = 0
        latent_state = self.dynamics_predictor(hidden_state)
        while self.continue_predictor(hidden_state, latent_state):
            last_hidden_state = hidden_state
            last_latent_state = latent_state
        
        return 0