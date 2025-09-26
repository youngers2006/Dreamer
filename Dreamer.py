import torch
import torch.nn as nn
import torch.optim as optim
from WorldModel import WorldModel
from Agent import Agent
from Buffer import Buffer

class Dreamer(nn.Module):
    def __init__(self, world_model_lr, actor_critic_lr, horizon):
        self.world_model = WorldModel()
        self.agent = Agent()
        self.buffer = Buffer()
        self.horizon = horizon

    def dream_episodes(self, starting_latent_state_batch, starting_hidden_state_batch):
        def dream_episode(starting_latent_state, starting_hidden_state):
            hidden_states = [starting_hidden_state]
            latent_states = [starting_latent_state]
            rewards = []
            actions = []
            continues_ = []
            a_mus = []
            a_sigmas = []
            for _ in range(self.horizon):
                action, a_mu, a_sigma = self.agent.actor.act(hidden_state, latent_state)
                hidden_state_, latent_state_, reward, continue_ = self.world_model.imagine_step(hidden_state, latent_state, action)
                hidden_states.append(hidden_state) ; latent_states.append(latent_state)
                rewards.append(reward) ; actions.append(action) ; continues_.append(continue_)
                a_mus.append(a_mu) ; a_sigmas.append(a_sigma)
                hidden_state = hidden_state_ ; latent_state = latent_state_
            latent_states = torch.stack(latent_states[:-1], dim=0)
            hidden_states = torch.stack(hidden_states[:-1], dim=0)
            actions = torch.stack(actions, dim=0)
            rewards = torch.stack(rewards, dim=0)
            continues_ = torch.stack(continues_, dim=0)
            a_mus = torch.stack(a_mus, dim=0)
            a_sigmas = torch.stack(a_sigmas, dim=0)
            return latent_states, hidden_states, actions, rewards, continues_, a_mus, a_sigmas
    
        dream_batch_fn = torch.vmap(dream_episode, in_dims=(0,0))
        z_batch_seq, h_batch_seq, a_batch_seq, rewards_batch_seq, continues_batch_seq, a_mu_batch_seq, a_sigma_batch_seq = dream_batch_fn(
            starting_latent_state_batch,
            starting_hidden_state_batch
        ) 
        return z_batch_seq, h_batch_seq, a_batch_seq, rewards_batch_seq, continues_batch_seq, a_mu_batch_seq, a_sigma_batch_seq
    
    def train_world_model(self):
        pass

    def warm_start_generator(self, observation_seq_batch, action_seq_batch, sequence_length):
        hidden_batch = torch.zeros(self.batch_size, self.hidden_dims, dtype=torch.float32, device=self.device)
        latent_batch, _, _ = self.world_model.encoder.encode(hidden_batch, observation_seq_batch[:, 0, :])
        warmup_length = sequence_length // 2
        for t in range(1, warmup_length):
            latent_batch, hidden_batch, _, _ = self.world_model.observe_step(
                latent_batch, 
                hidden_batch, 
                action_seq_batch[:,t-1,:], 
                observation_seq_batch[:, t, :]
            )
        return latent_batch, hidden_batch
    
    def train_Agent(self):
        for epoch in range(self.epochs):
            observation_seq_batch, action_seq_batch, _, _, sequence_length = self.buffer.sample_sequences()
            initial_latent_batch, initial_hidden_batch = self.warm_start_generator(observation_seq_batch, action_seq_batch, sequence_length)
            latent_seq_batch_dream, hidden_seq_batch_dream, action_seq_batch_dream, reward_seq_batch_dream, continue_seq_batch_dream, a_mu_batch_seq, a_sigma_batch_seq = self.dream_episodes(
                initial_latent_batch,
                initial_hidden_batch
            )
            self.world_model
            self.agent.train_step(latent_seq_batch_dream, hidden_seq_batch_dream, R_batch_seq, action_batch_seq, a_mu_batch_seq, a_sigma_batch_seq)



