import torch
import torch.nn as nn
from torch.distributions import Normal

class Agent(nn.Module):
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
            AC_lr, 
            AC_betas, 
            AC_eps,
            *, 
            device='cpu'
        ):
        self.device = device
        self.actor = Actor(action_dim, latent_column_dim, latent_row_dim, hidden_state_dim, HL_A1, HL_A2, device=device)
        self.critic = Critic(latent_column_dim, latent_row_dim, hidden_state_dim, HL_C1, HL_C2, device=device)

        self.optimiser = torch.optim.AdamW(
            params=self.parameters(),
            lr=AC_lr,
            betas=(AC_betas[0], AC_betas[1]),
            eps=AC_eps
        )

    def train_step(self, ):
        pass

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



