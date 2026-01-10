import torch
import torch.nn as nn
from DreamerUtils import symexp

class DynamicsPredictor(nn.Module):
    """
    Takes the current hidden state of the sequence model and predicts the encoded state
    """
    def __init__(self, latent_num_rows, latent_num_columns, hidden_state_size, hidden_L1, hidden_L2, device):
        super().__init__()
        self.latent_num_rows = latent_num_rows
        self.latent_num_columns = latent_num_columns
        self.latent_size = latent_num_rows * latent_num_columns
        self.device = device
        self.logit_net = nn.Sequential(
            nn.Linear(in_features=hidden_state_size, out_features=hidden_L1, device=device),
            nn.LayerNorm(hidden_L1, device=device),
            nn.SiLU(),
            nn.Linear(in_features=hidden_L1, out_features=hidden_L2, device=device),
            nn.LayerNorm(hidden_L2, device=device),
            nn.SiLU(),
            nn.Linear(in_features=hidden_L2, out_features=self.latent_size, device=device)
        )

    def forward(self, x):
        logits = self.logit_net(x)
        B, S, _ = logits.shape
        logits = logits.view(B, S, self.latent_num_rows, self.latent_num_columns)
        return logits
    
    def predict(self, hidden_state: torch.tensor):
        logits = self.forward(hidden_state)
        probs = torch.softmax(logits, dim=-1)
        uniform = (1.0 / self.latent_num_columns)
        probs = 0.99 * probs + 0.01 * uniform
        dist = torch.distributions.Categorical(probs=probs)
        sample_idx = dist.sample()
        latent_state_OH = torch.nn.functional.one_hot(sample_idx, num_classes=self.latent_num_columns).float()
        latent_state = latent_state_OH + probs - probs.detach()
        return latent_state, logits
    
class RewardPredictor(nn.Module):
    """ 
    Takes the current hidden state of the sequence model and the latent state and predicts the reward
    """
    def __init__(self, latent_num_rows, latent_num_columns, hidden_state_size, hidden_L1, hidden_L2, num_buckets=255, device='cpu'):
        super().__init__()
        self.latent_size = latent_num_rows * latent_num_columns
        self.buckets = num_buckets
        self.device = device
        self.flatten = nn.Flatten(start_dim=2)
        self.logit_net = nn.Sequential(
            nn.Linear(in_features=hidden_state_size + self.latent_size, out_features=hidden_L1, device=device),
            nn.LayerNorm(hidden_L1, device=device),
            nn.SiLU(),
            nn.Linear(in_features=hidden_L1, out_features=hidden_L2, device=device),
            nn.LayerNorm(hidden_L2, device=device),
            nn.SiLU(),
            nn.Linear(in_features=hidden_L2, out_features=num_buckets, device=device)
        )
        buckets = torch.linspace(-20.0, 20.0, num_buckets, device=device)
        self.register_buffer('buckets_rew', buckets)
    
    def forward(self, hidden, latent):
        latent = self.flatten(latent)
        input = torch.cat([hidden, latent], dim=-1)
        logits = self.logit_net(input)
        return logits
    
    def predict(self, hidden_state: torch.tensor, latent_state: torch.tensor):
        logits = self.forward(hidden_state, latent_state)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        symlog_reward = torch.sum(probs * self.buckets_rew, dim=-1, keepdim=True)
        return symexp(symlog_reward)
    
class ContinuePredictor(nn.Module):
    """
    Takes the current hidden state of the sequence model and the latent state and predicts if imagined predicted episode should continue
    """
    def __init__(self, latent_num_rows, latent_num_columns, hidden_state_size, hidden_L1, hidden_L2, device):
        super().__init__()
        self.latent_size = latent_num_rows * latent_num_columns
        self.device = device
        self.flatten = nn.Flatten(start_dim=2)
        self.logit_generator = nn.Sequential(
            nn.Linear(in_features=hidden_state_size + self.latent_size, out_features=hidden_L1, device=device),
            nn.LayerNorm(hidden_L1, device=device),
            nn.SiLU(),
            nn.Linear(in_features=hidden_L1, out_features=hidden_L2, device=device),
            nn.LayerNorm(hidden_L2, device=device),
            nn.SiLU(),
            nn.Linear(in_features=hidden_L2, out_features=1, device=device)
        )

    def forward(self, hidden, latent):
        latent = self.flatten(latent)
        input = torch.cat([hidden, latent], dim=-1)
        logit = self.logit_generator(input)
        probability = torch.sigmoid(logit)
        return probability, logit
    
    def predict(self, hidden_state: torch.tensor, latent_state: torch.tensor):
        probability, _ = self.forward(hidden_state, latent_state)
        continue_ = (probability >= 0.5)
        return continue_