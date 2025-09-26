import torch
import torch.nn as nn
import torch.optim as optim
from WorldModel import WorldModel
from Agent import Agent
from Buffer import Buffer

class Dreamer(nn.Module):
    def __init__(self, world_model_lr, actor_critic_lr):
        self.world_model = WorldModel()
        self.agent = Agent()
        self.buffer = Buffer()

    def train_world_model(self):
        pass

    def train_Agent(self):
        pass