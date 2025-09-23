import torch
import torch.nn as nn
from VariationalAutoEncoder import Decoder, Encoder
from DynamicsPredictors import DynamicsPredictor, RewardPredictor, ContinuePredictor
from SequenceModel import SequenceModel

class WorldModel(nn.Module):
    def __init__(self, hidden_dims, latent_dims, optimiser, device='cpu'):
        self.latent_dims = latent_dims
        self.hidden_dims = hidden_dims
        self.state_dims = latent_dims + hidden_dims # st = [ht, zt]
        self.optimiser = optimiser

        self.encoder = Encoder()
        self.sequence_model = SequenceModel()
        self.dynamics_predictor = DynamicsPredictor()
        self.reward_predictor = RewardPredictor()
        self.continue_predictor = ContinuePredictor()
        self.decoder = Decoder()
    
    def imagine_episode(self):
        hidden_state = 0
        latent_state = self.dynamics_predictor(hidden_state)
        while self.continue_predictor(hidden_state, latent_state):
            last_hidden_state = hidden_state
            last_latent_state = latent_state

            


        return state, hidden_state
    
    def train(self):

