import torch
import torch.nn as nn
from VariationalAutoEncoder import Decoder, Encoder

class WorldModel(nn.Module):
    def __init__(self, hidden_dims, latent_dims, device='cpu'):
        self.latent_dims = latent_dims
        self.hidden_dims = hidden_dims
        self.state_dims = latent_dims + hidden_dims # st = [ht, zt]

        self.GRU = nn.GRU(input_size=latent_dims, hidden_size=hidden_dims, device=device)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.policy_base = Policy()
    
    def 
