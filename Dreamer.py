import torch
import torch.nn as nn
import torch.optim as optim
from WorldModel import WorldModel
from Agent import Actor, Critic

class Dreamer(nn.Module):
    def __init__(self, world_model_lr, actor_critic_lr):
        self.world_model = WorldModel()
        self.actor = Actor()
        self.critic = Critic()

    def learn_world_model(self):
        pass

    def 