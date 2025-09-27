import torch
import torch.nn as nn
from torch.distributions import Normal
from DreamerUtils import gaussian_log_probability

class Agent(nn.Module): # batched sequence (batch_size, sequence_length, features*)
    def __init__(
            self, 
            action_dim,
            latent_column_dim,
            latent_row_dim,
            hidden_state_dim,
            HL_A1,
            HL_A2,
            HL_C1,
            HL_C2, 
            A_lr, 
            A_betas, 
            A_eps,
            C_lr, 
            C_betas, 
            C_eps,
            nu,
            lambda_,
            gamma,
            *, 
            device='cpu'
        ):
        self.device = device
        self.actor = Actor(action_dim, latent_column_dim, latent_row_dim, hidden_state_dim, HL_A1, HL_A2, device=device)
        self.critic = Critic(latent_column_dim, latent_row_dim, hidden_state_dim, HL_C1, HL_C2, device=device)

        self.nu = nu
        self.lambda_ = lambda_
        self.gamma = gamma

        self.actor_optimiser = torch.optim.AdamW(
            params=self.parameters(),
            lr=A_lr,
            betas=(A_betas[0], A_betas[1]),
            eps=A_eps
        )
        self.critic_optimiser = torch.optim.AdamW(
            params=self.parameters(),
            lr=C_lr,
            betas=(C_betas[0], C_betas[1]),
            eps=C_eps
        )

    def train_step(self, z_batch_seq, h_batch_seq, reward_batch_seq, continue_batch_seq, action_batch_seq, a_mu_batch_seq, a_sigma_batch_seq):
        R_lambda_batch_seq = self.compute_batched_R_lambda_returns(
                h_batch_seq,
                z_batch_seq,
                reward_batch_seq,
                continue_batch_seq,
                continue_batch_seq.shape[1]
            )
        value_batched_seq = self.critic.forward(z_batch_seq, h_batch_seq)
        advantage_batched_seq = R_lambda_batch_seq - value_batched_seq

        a_dist_batch_seq = Normal(loc=a_mu_batch_seq, scale=a_sigma_batch_seq)
        log_prob_batch_seq = a_dist_batch_seq.log_prob(action_batch_seq)
        actor_entropy_batched_seq = a_dist_batch_seq.entropy()
        loss_batched_seq_actor = log_prob_batch_seq * advantage_batched_seq.detach() + self.nu.detach() * actor_entropy_batched_seq.detach()
        loss_actor = torch.mean(loss_batched_seq_actor, keepdim=False)
        loss_actor *= -1.0

        loss_batched_seq_critic = (0.5) * torch.square(value_batched_seq - R_lambda_batch_seq.detach())
        loss_critic = torch.mean(loss_batched_seq_critic, keepdim=False)

        self.actor_optimiser.zero_grad()
        loss_actor.backward()
        self.actor_optimiser.step()

        self.critic_optimiser.zero_grad()
        loss_critic.backward()
        self.critic_optimiser.step()
        return loss_actor, loss_critic
    
    def compute_batched_R_lambda_returns(self, hidden_state_batched_seq, latent_state_batched_seq, reward_batched_seq, continue_batched_seq, seq_length):
        def compute_R_lambda_returns(hidden_state_seq, latent_state_seq, reward_seq, continue_seq, seq_length):
            value_estimate_seq = self.critic(hidden_state_seq, latent_state_seq).detach()
            R_lambda_seq = [value_estimate_seq[-1]]
            for t in reversed(range(seq_length-1)):
                R_lambda = reward_seq[t] + self.gamma * continue_seq[t] * ((1 - self.lambda_) * value_estimate_seq[t+1] + self.lambda_ * R_lambda_seq[0])
                R_lambda_seq.insert(0, R_lambda)
            R_lambda_seq = torch.stack(R_lambda_seq, dim=0)
            return R_lambda_seq
        
        R_lambda_batched_seq = torch.vmap(compute_R_lambda_returns, in_dims=(0,0,0,0,None))(
            hidden_state_batched_seq, 
            latent_state_batched_seq, 
            reward_batched_seq, 
            continue_batched_seq, 
            seq_length
        )
        return R_lambda_batched_seq
        
class Actor(nn.Module):
    def __init__(self, action_dim, latent_column_dim, latent_row_dim, hidden_state_dim, hidden_layer_num_nodes_1, hidden_layer_num_nodes_2,*, device='cpu'):
        super().__init__()
        self.flatten = nn.Flatten()
        self.base_net = nn.Sequential(
            nn.Linear(in_features=latent_row_dim * latent_column_dim + hidden_state_dim, out_features=hidden_layer_num_nodes_1, device=device),
            nn.SiLU(),
            nn.Linear(in_features=hidden_layer_num_nodes_2, out_features=hidden_layer_num_nodes_2, device=device),
            nn.SiLU()
        )
        self.mu_head = nn.Linear(in_features=hidden_layer_num_nodes_2, out_features=action_dim, device=device)
        self.log_sig_head = nn.Linear(in_features=hidden_layer_num_nodes_2, out_features=action_dim, device=device)

    def forward(self, ht, zt):
        flattened_zt = self.flatten(zt)
        st = torch.cat([ht, flattened_zt], dim=-1)
        base_result = self.base_net(st)
        mu = self.mu_head(base_result)
        log_sig = self.log_sig_head(base_result)
        sigma = torch.log(log_sig)
        return mu, sigma
    
    def act(self, ht, zt):
        mu, sigma = self.forward(ht, zt)
        action_dist = Normal(loc=mu, scale=sigma)
        action = action_dist.rsample()
        return action, mu, sigma

class Critic(nn.Module):
    def __init__(self, latent_column_dim, latent_row_dim, hidden_state_dim, hidden_layer_num_nodes_1, hidden_layer_num_nodes_2,*, device='cpu'):
        super().__init__()
        self.latent_row_dim = latent_row_dim
        self.latent_column_dim = latent_column_dim
        self.flatten = nn.Flatten()
        self.value_net = nn.Sequential(
            nn.Linear(in_features=latent_column_dim * latent_row_dim + hidden_state_dim, out_features=hidden_layer_num_nodes_1, device=device),
            nn.SiLU(),
            nn.Linear(in_features=hidden_layer_num_nodes_1, out_features=hidden_layer_num_nodes_2, device=device),
            nn.SiLU(),
            nn.Linear(in_features=hidden_layer_num_nodes_2, out_features=1, device=device)
        )

    def forward(self, ht, zt):
        flattened_zt = self.flatten(zt)
        st = torch.cat([ht, flattened_zt], dim=-1)
        return self.value_net(st)



