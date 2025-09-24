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

        combined_params = list(self.actor.parameters(), self.critic.parameters())
        self.world_model_optimiser = optim.AdamW(params=self.world_model.parameters(), lr=world_model_lr)
        self.actor_critic_optimiser = optim.AdamW(params=combined_params, lr=actor_critic_lr)