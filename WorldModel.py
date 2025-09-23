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

        def encode_observation(self, observation):
            return self.encoder.encode()
    
        def imagine_step(self, hidden_state, latent_state, action):
            next_hidden_state = self.sequence_model(hidden_state, latent_state, action)
            next_latent_state = self.dynamics_predictor.predict(next_hidden_state)
            reward = self.reward_predictor.predict(next_hidden_state, next_latent_state)
            continue_ = self.continue_predictor.predict(next_hidden_state, next_latent_state)
            return hidden_state, latent_state, reward, continue_

