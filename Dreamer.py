import torch
import torch.nn as nn
from WorldModel import WorldModel
from Agent import Actor, Critic

class Dreamer(nn.Module):
    def __init__(self):
        self.world_model = WorldModel()
        self.actor = Actor()
        self.critic = Critic()
    