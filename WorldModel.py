import torch
import torch.nn as nn
from VariationalAutoEncoder import Decoder, Encoder
from DynamicsPredictors import DynamicsPredictor, RewardPredictor, ContinuePredictor
from SequenceModel import SequenceModel
import torch.distributions as distributions
import torch.optim as optim

class WorldModel(nn.Module):
    def __init__(self, hidden_dims, latent_dims, observation_dims, training_horizon, batch_size, optimiser, beta_pred, beta_dyn, beta_reg, device='cpu'):
        self.latent_dims = latent_dims
        self.hidden_dims = hidden_dims
        self.observation_dim_x, self.observation_dim_y = observation_dims
        self.state_dims = latent_dims + hidden_dims # st = [ht, zt]
        self.horizon = training_horizon

        self.optimiser = optimiser
        self.beta_pred = beta_pred
        self.beta_dyn = beta_dyn
        self.beta_reg = beta_reg
        self.batch_size = batch_size

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
        latent_state, latent_mu, latent_sigma = self.encode_observation(observation, hidden_state)
        return latent_state, hidden_state, latent_mu, latent_sigma
    
    def training_step(
            self, 
            observation_sequences: torch.tensor, 
            action_sequences: torch.tensor, 
            reward_sequences: torch.tensor, 
            continue_sequences: torch.tensor
    ): # (sequence length, batch size, features)
        hidden_state_sequences = torch.zeros(self.horizon, self.batch_size, self.hidden_dims)
        latent_state_sequences = torch.zeros(self.horizon, self.batch_size, self.latent_dims)

        decoder_pred_val = torch.zeros(self.horizon, self.batch_size, 1)
        reward_pred_val = torch.zeros(self.horizon, self.batch_size, 1)
        continue_pred_val = torch.zeros(self.horizon, self.batch_size, 1)

        target_dyn = torch.zeros(self.horizon, self.batch_size, 2)
        target_reg = torch.zeros(self.horizon, self.batch_size, 2)
        pred_dyn_dist = torch.zeros(self.horizon, self.batch_size, 2)
        pred_reg_dist = torch.zeros(self.horizon, self.batch_size, 2)

        for t in range(1, self.horizon):
            latent_state, hidden_state, enc_mu, enc_sig = self.observe_step(
                latent_state_sequences[t-1], 
                hidden_state_sequences[t-1], 
                action_sequences[t-1],
                observation_sequences[t-1]
            )
            hidden_state_sequences[t] = hidden_state ; latent_state_sequences[t] = latent_state
    
            dec_mu, dec_sig = self.decoder(hidden_state, latent_state)
            cont_prob, _ = self.continue_predictor(hidden_state, latent_state)
            rew_mu, rew_sig = self.reward_predictor(hidden_state, latent_state) 
            dyn_mu, dyn_sig = self.dynamics_predictor(hidden_state, latent_state)

            target_dyn[t,1] = enc_mu ; target_dyn[t,2] = enc_sig
            target_reg[t,1] = dyn_mu ; target_reg[t,2] = dyn_sig
            pred_dyn_dist[t,1] = dyn_mu ; pred_dyn_dist[t,2] = dyn_sig
            pred_reg_dist[t,1] = enc_mu ; pred_reg_dist[t,2] = enc_sig

            decoder_dist = distributions.Normal(loc=dec_mu, scale=dec_sig)
            reward_dist = distributions.Normal(loc=rew_mu, scale=rew_sig)

            decoder_pred_val[t] = decoder_dist.log_prob(observation_sequences[t])
            reward_pred_val[t] = reward_dist.log_prob(reward_sequences[t])
            continue_correct_prob = cont_prob * continue_sequences[t] + (1 - cont_prob) * continue_sequences[t]
            continue_pred_val[t] = torch.log(continue_correct_prob)

        loss_pred = - decoder_pred_val - reward_pred_val - continue_pred_val

        pred_dyn_dist_ = distributions.Normal(loc=pred_dyn_dist[t,1], scale=pred_dyn_dist[t,2])
        target_dyn_ = distributions.Normal(loc=target_dyn[t,1], scale=target_dyn[t,2])
        pred_reg_dist_ = distributions.Normal(loc=pred_reg_dist[t,1], scale=pred_reg_dist[t,2])
        target_reg_ = distributions.Normal(loc=target_reg[t,1], scale=target_reg[t,2])

        loss_dyn = torch.max(1, distributions.kl.kl_divergence(pred_dyn_dist_, target_dyn_.detach()))
        loss_reg = torch.max(1, distributions.kl.kl_divergence(pred_reg_dist_, target_reg_.detach()))

        total_loss = torch.sum(self.beta_pred * loss_pred + self.beta_dyn * loss_dyn + self.beta_reg * loss_reg)

        self.optimiser.reset()
        total_loss.backward()
        self.optimiser.step()
        



