import torch
import torch.nn as nn

class DynamicsPredictor(nn.Module):
    """
    Takes the current hidden state of the sequence model and predicts the encoded state
    """
    def __init__(self, latent_size, hidden_state_size, hidden_L1, hidden_L2):
        self.latent_size = latent_size
        self.base_network = nn.Sequential(
            nn.Linear(in_features=hidden_state_size, out_features=hidden_L1),
            nn.SiLU(),
            nn.Linear(in_features=hidden_L1, out_features=hidden_L2),
            nn.SiLU()
        )
        self.mu_head = nn.Linear(in_features=hidden_L2, out_features=latent_size)
        self.log_sig_head = nn.Linear(in_features=hidden_L2, out_features=latent_size)

    def forward(self, x):
        x = self.base_network(x)
        mu = self.mu_head(x)
        log_sig = self.log_sig_head(x)
        sigma = torch.exp(log_sig)
        return mu, sigma
    
    def predict(self, hidden_state: torch.tensor):
        mu, sigma = self.forward(hidden_state)
        dist = torch.distributions.Normal(loc=mu, scale=sigma)
        next_latent_state = dist.rsample(self.latent_size)
        return next_latent_state
    
class RewardPredictor(nn.Module):
    """
    Takes the current hidden state of the sequence model and the latent state and predicts the reward
    """
    def __init__(self, latent_size, hidden_state_size, hidden_L1, hidden_L2):
        self.latent_size = latent_size
        self.base_network = nn.Sequential(
            nn.Linear(in_features=hidden_state_size + latent_size, out_features=hidden_L1),
            nn.SiLU(),
            nn.Linear(in_features=hidden_L1, out_features=hidden_L2),
            nn.SiLU()
        )
        self.mu_head = nn.Linear(in_features=hidden_L2, out_features=1)
        self.log_sig_head = nn.Linear(in_features=hidden_L2, out_features=1)

    def forward(self, x):
        x = self.base_network(x)
        mu = self.mu_head(x)
        log_sig = self.log_sig_head(x)
        sigma = torch.exp(log_sig)
        return mu, sigma
    
    def predict(self, hidden_state: torch.tensor, latent_state: torch.tensor):
        state = torch.cat(hidden_state, latent_state)
        mu, sigma = self.forward(state)
        dist = torch.distributions.Normal(loc=mu, scale=sigma)
        next_state = dist.rsample(1)
        return next_state
    
class ContinuePredictor(nn.Module):
    """
    Takes the current hidden state of the sequence model and the latent state and predicts if imagined predicted episode should continue
    """
    def __init__(self, latent_size, hidden_state_size, hidden_L1, hidden_L2):
        self.latent_size = latent_size
        self.logit_generator = nn.Sequential(
            nn.Linear(in_features=hidden_state_size + latent_size, out_features=hidden_L1),
            nn.SiLU(),
            nn.Linear(in_features=hidden_L1, out_features=hidden_L2),
            nn.SiLU(),
            nn.Linear(in_features=hidden_L2, out_features=1)
        )

    def forward(self, x):
        logit = self.logit_generator(x)
        return logit
    
    def predict(self, hidden_state: torch.tensor, latent_state: torch.tensor):
        state = torch.cat(hidden_state, latent_state)
        logit = self.forward(state)
        probability = torch.sigmoid(logit)
        continue_ = (probability <= 0.05)
        return continue_
    
