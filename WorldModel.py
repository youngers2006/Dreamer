import torch
import torch.nn as nn
from VariationalAutoEncoder import Decoder, Encoder
from DynamicsPredictors import DynamicsPredictor, RewardPredictor, ContinuePredictor
from SequenceModel import SequenceModel
import torch.distributions as distributions

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

    def encode_observation(self, observation, hidden_state):
        return self.encoder.encode(observation, hidden_state)
        
    def decode_latent_state(self, latent_state, hidden_state):
        return self.decoder.decode(latent_state, hidden_state)
    
    def imagine_step(self, hidden_state, latent_state, action):
        next_hidden_state = self.sequence_model(hidden_state, latent_state, action)
        next_latent_state = self.dynamics_predictor.predict(next_hidden_state)
        next_reward = self.reward_predictor.predict(next_hidden_state, next_latent_state)
        continue_ = self.continue_predictor.predict(next_hidden_state, next_latent_state)
        return next_hidden_state, next_latent_state, next_reward, continue_
    
    def observe_step(self, last_latent, last_hidden, last_action, observation):
        hidden_state = self.sequence_model.forward(last_latent, last_hidden, last_action)
        latent_state = self.encode_observation(observation, hidden_state)
        return latent_state, hidden_state
    
    def training_step(self, observation_sequences: torch.tensor, action_sequences: torch.tensor, reward_sequences: torch.tensor, continue_sequences: torch.tensor):
        dec_mu, dec_sig = self.decoder(hi) ; rew_mu, rew_sig = self.reward_predictor() ; cont_mu, cont_sig = self.continue_predictor()
        enc_mu, enc_sig = self.encoder() ; dyn_mu, dyn_sig = self.dynamics_predictor()
        decoder_dist = distributions.Normal(loc=dec_mu, scale=dec_sig)
        reward_dist = distributions.Normal(loc=rew_mu, scale=rew_sig)
        continue_dist = distributions.Normal(loc=cont_mu, scale=cont_sig)
        encoder_dist = distributions.Normal(loc=enc_mu, scale=enc_sig)
        dynamics_dist = distributions.Normal(loc=dyn_mu, scale=dyn_sig)

        target_dyn = encoder_dist.log_prob(latent_state)
        target_reg = dynamics_dist.log_prob(latent_state)

        loss_pred = - decoder_dist.log_prob(observation) - reward_dist.log_prob(reward) - continue_dist.log_prob(continue_)
        loss_dyn = torch.max(1, torch.kl_div(dynamics_dist.log_prob(latent_state), target_dyn.detach()))
        loss_reg = torch.max(1, torch.kl_div(encoder_dist.log_prob(latent_state), target_reg.detach()))
        total_loss = torch.sum(self.beta_pred * loss_pred + self.beta_dyn * loss_dyn + self.beta_reg * loss_reg)

        self.optimiser.reset()
        total_loss.backward()
        self.optimiser.step()
        



