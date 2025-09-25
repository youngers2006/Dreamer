import torch
import torch.nn as nn
from VariationalAutoEncoder import Decoder, Encoder
from DynamicsPredictors import DynamicsPredictor, RewardPredictor, ContinuePredictor
from SequenceModel import SequenceModel
import torch.distributions as distributions
from torch.distributions import Normal, Bernoulli
import torch.optim as optim
from DreamerUtils import gaussian_log_probability, kullback_leibler_divergence_between_gaussians, bernoulli_log_probability

class WorldModel(nn.Module):
    def __init__(
            self, 
            hidden_dims, 
            latent_dims, 
            observation_dims, 
            action_dims, 
            training_horizon, 
            batch_size, 
            optimiser, 
            beta_pred, 
            beta_dyn, 
            beta_rep, 
            num_encoder_filters_1, 
            num_encoder_filters_2,
            encoder_hidden_layer_nodes,
            num_decoder_filters_1, 
            num_decoder_filters_2, 
            decoder_hidden_layer_nodes,
            dyn_pred_hidden_num_nodes_1, 
            dyn_pred_hidden_num_nodes_2,
            rew_pred_hidden_num_nodes_1, 
            rew_pred_hidden_num_nodes_2, 
            cont_pred_hidden_num_nodes_1, 
            cont_pred_hidden_num_nodes_2,
            device='cpu'
        ):
        super().__init__()
        self.latent_dims = latent_dims
        self.hidden_dims = hidden_dims
        self.action_dims = action_dims
        self.observation_dim_x, self.observation_dim_y = observation_dims
        self.state_dims = latent_dims + hidden_dims # st = [ht, zt]
        self.horizon = training_horizon

        self.optimiser = optimiser
        self.beta_pred = beta_pred
        self.beta_dyn = beta_dyn
        self.beta_rep = beta_rep
        self.batch_size = batch_size

        self.encoder = Encoder(latent_dims, num_encoder_filters_1, num_encoder_filters_2, encoder_hidden_layer_nodes, device=device)
        self.sequence_model = SequenceModel(latent_dims, hidden_dims, action_dims, num_layers=1, device=device)
        self.dynamics_predictor = DynamicsPredictor(latent_dims, hidden_dims, dyn_pred_hidden_num_nodes_1, dyn_pred_hidden_num_nodes_2, device)
        self.reward_predictor = RewardPredictor(latent_dims, hidden_dims, rew_pred_hidden_num_nodes_1, rew_pred_hidden_num_nodes_2, device=device)
        self.continue_predictor = ContinuePredictor(latent_dims, hidden_dims, cont_pred_hidden_num_nodes_1, cont_pred_hidden_num_nodes_2, device=device)
        self.decoder = Decoder(latent_dims, observation_dims, hidden_dims, num_decoder_filters_1, num_decoder_filters_2, decoder_hidden_layer_nodes, device=device)
        self.device = device

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
    
    def observe_step(self, last_latent, last_hidden, last_action, observation) -> tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        hidden_state = self.sequence_model.forward(last_latent, last_hidden, last_action)
        latent_state, latent_mu, latent_sigma = self.encode_observation(observation, hidden_state)
        return latent_state, hidden_state, latent_mu, latent_sigma
    
    def unroll_model(
            self,
            observation_sequence_batch: torch.tensor, 
            action_sequence_batch: torch.tensor, 
            reward_sequence_batch: torch.tensor, 
            continue_sequence_batch: torch.tensor
        ):
        def single_sequence_unroll(observation_sequence, action_sequence, reward_sequence, continue_sequence, init_hidden_state_sequence, init_latent_state_sequence):
            posterior_mu = torch.zeros(self.horizon, self.latent_dims, dtype=torch.float32, device=self.device)
            posterior_sigma = torch.zeros(self.horizon, self.latent_dims, dtype=torch.float32, device=self.device)
            prior_mu = torch.zeros(self.horizon, self.latent_dims, dtype=torch.float32, device=self.device)
            prior_sigma = torch.zeros(self.horizon, self.latent_dims, dtype=torch.float32, device=self.device)
            obs_likelyhood_seq = torch.zeros(self.horizon, 1, dtype=torch.float32, device=self.device)
            rew_likelyhood_seq = torch.zeros(self.horizon, 1, dtype=torch.float32, device=self.device) 
            cont_likelyhood_seq = torch.zeros(self.horizon, 1, dtype=torch.float32, device=self.device)

            for t in range(1, self.horizon):
                posterior_latent, hidden_state, posterior_latent_mu, posterior_latent_sig = self.observe_step(
                    init_latent_state_sequence[t-1],
                    init_hidden_state_sequence[t-1],
                    action_sequence[t-1],
                    observation_sequence[t-1]
                )
                prior_latent_mu, prior_latent_sig = self.dynamics_predictor(hidden_state, posterior_latent)
                dec_mu, dec_sig = self.decoder(hidden_state, posterior_latent)
                rew_mu, rew_sig = self.reward_predictor(hidden_state, posterior_latent) 
                cont_prob, _ = self.continue_predictor(hidden_state, posterior_latent)

                observation_log_likelyhood = gaussian_log_probability(observation_sequence[t], dec_mu, dec_sig).sum()
                reward_log_likelyhood = gaussian_log_probability(reward_sequence[t], rew_mu, rew_sig)
                continue_log_likelyhood = bernoulli_log_probability(cont_prob, continue_sequence[t])

                posterior_mu[t] = posterior_latent_mu ; posterior_sigma[t] = posterior_latent_sig
                prior_mu[t] = prior_latent_mu ; prior_sigma[t] = prior_latent_sig
                obs_likelyhood_seq[t] = observation_log_likelyhood
                rew_likelyhood_seq[t] = reward_log_likelyhood
                cont_likelyhood_seq[t] = continue_log_likelyhood
            
            return prior_mu, prior_sigma, posterior_mu, posterior_sigma, obs_likelyhood_seq, rew_likelyhood_seq, cont_likelyhood_seq

        init_hidden_state_sequence = torch.zeros(self.horizon, self.hidden_dims, dtype=torch.float32, device=self.device)
        init_latent_state_sequence = torch.zeros(self.horizon, self.latent_dims, dtype=torch.float32, device=self.device)

        batched_sequence_unroll = torch.vmap(single_sequence_unroll, in_dims=(0,0,0,0,None,None))
        prior_mu, prior_sigma, posterior_mu, posterior_sigma, obs_log_lh, rew_log_lh, cont_log_lh = batched_sequence_unroll(
            observation_sequence_batch,
            action_sequence_batch,
            reward_sequence_batch,
            continue_sequence_batch,
            init_hidden_state_sequence, 
            init_latent_state_sequence
        )
        return prior_mu, prior_sigma, posterior_mu, posterior_sigma, obs_log_lh, rew_log_lh, cont_log_lh

    def training_step(
            self, 
            observation_sequences: torch.tensor, 
            action_sequences: torch.tensor, 
            reward_sequences: torch.tensor, 
            continue_sequences: torch.tensor
    ): # (sequence length, batch size, features)
        prior_mu, prior_sigma, posterior_mu, posterior_sigma, obs_log_lh, rew_log_lh, cont_log_lh = self.unroll_model(
            observation_sequences,
            action_sequences,
            reward_sequences,
            continue_sequences
        )

        Dkl_dyn = kullback_leibler_divergence_between_gaussians(posterior_mu.detach(), posterior_sigma.detach(), prior_mu, prior_sigma)
        Dkl_rep = kullback_leibler_divergence_between_gaussians(posterior_mu, posterior_sigma, prior_mu.detach(), prior_sigma.detach())
        loss_pred = - obs_log_lh - rew_log_lh - cont_log_lh
        loss_dyn = torch.max(1, Dkl_dyn, dim=0)
        loss_rep = torch.max(1, Dkl_rep, dim=0)
        combined_loss = self.beta_pred * loss_pred + self.beta_dyn * loss_dyn + self.beta_rep * loss_rep
        total_loss = combined_loss.mean()

        self.optimiser.reset()
        total_loss.backward()
        self.optimiser.step()

        return total_loss
        



