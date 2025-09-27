import torch
import torch.nn as nn

class SequenceModel(nn.Module):
    def __init__(self, latent_num_rows, latent_num_columns, hidden_dim, action_dim, *, num_layers=1, device='cpu'):
        super().__init__()
        self.latent_dim = latent_num_columns * latent_num_rows
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device

        self.flatten = nn.Flatten()
        self.GRU = nn.GRU(
            input_size=self.latent_dim + action_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=False,
            device=device
        )

    def forward(self, last_latent_state: torch.tensor, last_hidden_state: torch.tensor, last_action: torch.tensor):
        last_latent_state = self.flatten(last_latent_state)
        input = torch.cat((last_latent_state, last_action), dim=-1)
        input = input.unsqueeze(0)
        last_hidden_state = last_hidden_state.unsqueeze(0)
        _, hidden = self.GRU(input, last_hidden_state)
        return hidden.squeeze(0)

