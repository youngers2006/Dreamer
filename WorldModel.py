import torch
import torch.nn as nn
from VariationalAutoEncoder import Decoder, Encoder
from DynamicsPredictors import DynamicsPredictor, RewardPredictor, ContinuePredictor
from SequenceModel import SequenceModel

class WorldModel(nn.Module):
    def __init__(self, hidden_dims, latent_dims, optimiser, beta_pred, beta_dyn, beta_reg, device='cpu'):
        self.latent_dims = latent_dims
        self.hidden_dims = hidden_dims
        self.state_dims = latent_dims + hidden_dims # st = [ht, zt]

        self.optimiser = optimiser
        self.beta_pred = beta_pred
        self.beta_dyn = beta_dyn
        self.beta_reg = beta_reg

        self.encoder = Encoder()
        self.sequence_model = SequenceModel()
        self.dynamics_predictor = DynamicsPredictor()
        self.reward_predictor = RewardPredictor()
        self.continue_predictor = ContinuePredictor()
        self.decoder = Decoder()

    def encode_observation(self, observation):
        return self.encoder.encode(observation)
        
    def decode_latent_state(self, latent_state):
        return self.decoder.decode(latent_state)
    
    def imagine_step(self, hidden_state, latent_state, action):
        next_hidden_state = self.sequence_model(hidden_state, latent_state, action)
        next_latent_state = self.dynamics_predictor.predict(next_hidden_state)
        next_reward = self.reward_predictor.predict(next_hidden_state, next_latent_state)
        continue_ = self.continue_predictor.predict(next_hidden_state, next_latent_state)
        return next_hidden_state, next_latent_state, next_reward, continue_
    
    def observe_step(self):
        pass
    
    def training_step(self, observation_sequences: torch.tensor, action_sequences: torch.tensor, reward_sequences: torch.tensor, continue_sequences: torch.tensor):
        loss_pred = - torch.log() - torch.log() - torch.log()
        loss_dyn = torch.max(1, torch.kl_div())
        loss_reg = torch.max(torch.kl_div(), 1)
        total_loss = torch.sum(self.beta_pred * loss_pred + self.beta_dyn * loss_dyn + self.beta_reg * loss_reg)

        self.optimiser.reset()
        total_loss.backward()
        self.optimiser.step()
        



