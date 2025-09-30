import torch
import torch.nn as nn
from VariationalAutoEncoder import Decoder, Encoder
from DynamicsPredictors import DynamicsPredictor, RewardPredictor, ContinuePredictor
from SequenceModel import SequenceModel
import torch.distributions as distributions
from torch.distributions import Normal, Bernoulli
import torch.optim as optim
from DreamerUtils import gaussian_log_probability, to_twohot, bernoulli_log_probability

class WorldModel(nn.Module):
    def __init__(
            self, 
            hidden_dims, 
            latent_dims, 
            observation_dims, 
            action_dims, 
            training_horizon, 
            batch_size, 
            WM_lr,
            WM_betas,
            WM_eps,
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
            reward_buckets,
            cont_pred_hidden_num_nodes_1, 
            cont_pred_hidden_num_nodes_2,
            device='cpu'
        ):
        super().__init__()
        self.latent_num_rows, self.latent_num_columns = latent_dims
        self.hidden_dims = hidden_dims
        self.action_dims = action_dims
        self.observation_dim_x, self.observation_dim_y = observation_dims
        self.horizon = training_horizon
        self.buckets = reward_buckets

        self.optimiser = torch.optim.AdamW(
            self.parameters(), 
            lr=WM_lr, 
            betas=(WM_betas[0], WM_betas[1]), 
            eps=WM_eps
        )
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

    def imagine_step(self, hidden_state, latent_state, action):
        next_hidden_state = self.sequence_model(hidden_state, latent_state, action)
        next_latent_state, _ = self.dynamics_predictor.predict(next_hidden_state)
        next_reward = self.reward_predictor.predict(next_hidden_state, next_latent_state)
        continue_ = self.continue_predictor.predict(next_hidden_state, next_latent_state)
        return next_hidden_state, next_latent_state, next_reward, continue_
    
    def observe_step(self, last_latent, last_hidden, last_action, observation) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        hidden_state = self.sequence_model.forward(last_latent, last_hidden, last_action)
        latent_state, latent_logits = self.encoder.encode(hidden_state, observation)
        return latent_state, hidden_state, latent_logits
    
    def unroll_model(
            self,
            observation_sequence_batch: torch.tensor, 
            action_sequence_batch: torch.tensor, 
            reward_sequence_batch: torch.tensor, 
            continue_sequence_batch: torch.tensor
        ):
        def single_sequence_unroll(observation_sequence, action_sequence, reward_sequence, continue_sequence):
            posterior_logits = []
            prior_logits = []
            obs_likelyhood_seq = []
            rew_likelyhood_seq = [] 
            cont_likelyhood_seq = []

            hidden_state = torch.zeros(self.hidden_dims, device=self.device)
            posterior_latent, posterior_logits_t = self.encoder.encode(hidden_state, observation_sequence[0])

            for t in range(self.horizon):
                prev_action = action_sequence[t-1] if t > 0 else torch.zeros(self.action_dims, device=self.device)
                posterior_latent, hidden_state, posterior_logits_t = self.observe_step(
                    posterior_latent,
                    hidden_state,
                    prev_action,
                    observation_sequence[t]
                )
                reward_th = to_twohot(reward_sequence[t], self.buckets)

                prior_latent_logits = self.dynamics_predictor(hidden_state)
                dec_mu, dec_sig = self.decoder(hidden_state, posterior_latent)
                rew_logits = self.reward_predictor(hidden_state, posterior_latent) 
                cont_prob, _ = self.continue_predictor(hidden_state, posterior_latent)

                observation_log_likelyhood = gaussian_log_probability(observation_sequence[t], dec_mu, dec_sig).sum()
                continue_log_likelyhood = bernoulli_log_probability(cont_prob, continue_sequence[t])
                reward_log_probs = torch.nn.functional.log_softmax(rew_logits, dim=-1)
                reward_log_likelyhood = torch.sum(reward_th * reward_log_probs)

                posterior_logits.append(posterior_logits_t)
                prior_logits.append(prior_latent_logits)
                obs_likelyhood_seq.append(observation_log_likelyhood)
                rew_likelyhood_seq.append(reward_log_likelyhood)
                cont_likelyhood_seq.append(continue_log_likelyhood)
            
            prior_logits = torch.stack(prior_logits, dim=0)
            posterior_logits = torch.stack(posterior_logits, dim=0)
            obs_likelyhood_seq = torch.stack(obs_likelyhood_seq, dim=0)
            rew_likelyhood_seq = torch.stack(rew_likelyhood_seq, dim=0)
            cont_likelyhood_seq = torch.stack(cont_likelyhood_seq, dim=0)
            
            return prior_logits, posterior_logits, obs_likelyhood_seq, rew_likelyhood_seq, cont_likelyhood_seq

        batched_sequence_unroll = torch.vmap(single_sequence_unroll, in_dims=(0,0,0,0))
        prior_logits, posterior_logits, obs_log_lh, rew_log_lh, cont_log_lh = batched_sequence_unroll(
            observation_sequence_batch,
            action_sequence_batch,
            reward_sequence_batch,
            continue_sequence_batch
        )
        return prior_logits, posterior_logits, obs_log_lh, rew_log_lh, cont_log_lh

    def training_step(
            self, 
            observation_sequences: torch.tensor, 
            action_sequences: torch.tensor, 
            reward_sequences: torch.tensor, 
            continue_sequences: torch.tensor
        ): # (batch_size, sequence_length, features)
        prior_logits, posterior_logits, obs_log_lh, rew_log_lh, cont_log_lh = self.unroll_model(
            observation_sequences,
            action_sequences,
            reward_sequences,
            continue_sequences
        )
        prior_dist = torch.distributions.Categorical(logits=prior_logits)
        posterior_dist = torch.distributions.Categorical(logits=posterior_logits)
        Dkl_dyn = torch.distributions.kl.kl_divergence(posterior_dist.detach(), prior_dist).sum(dim=-1).mean()
        Dkl_rep = torch.distributions.kl.kl_divergence(posterior_dist, prior_dist.detach()).sum(dim=-1).mean()
        loss_pred_batch = - obs_log_lh - rew_log_lh - cont_log_lh
        loss_pred = loss_pred_batch.mean()
        loss_dyn = torch.max(torch.tensor(1.0, device=self.device), Dkl_dyn)
        loss_rep = torch.max(torch.tensor(1.0, device=self.device), Dkl_rep)
        total_loss = self.beta_pred * loss_pred + self.beta_dyn * loss_dyn + self.beta_rep * loss_rep

        self.optimiser.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 100.0)
        self.optimiser.step()

        return total_loss