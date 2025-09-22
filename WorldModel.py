import torch
import torch.nn as nn
from VariationalAutoEncoder import Decoder, Encoder

class WorldModel(nn.Module):
    def __init__(self, hidden_dims, latent_dims,  optimiser, device='cpu'):
        self.latent_dims = latent_dims
        self.hidden_dims = hidden_dims
        self.state_dims = latent_dims + hidden_dims # st = [ht, zt]
        self.optimiser = optimiser
        self.GRU = nn.GRU(input_size=latent_dims, hidden_size=hidden_dims, device=device)
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, observation, last_hidden):
        latent_state = self.encoder(observation)
        hidden_state = self.GRU(last_hidden)
        state = torch.cat(hidden_state, latent_state, dim=-1)
        return state, hidden_state
    
    def train(self):

