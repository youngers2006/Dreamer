import torch
import torch.nn as nn

class SequenceModel(nn.Module):
    def __init__(self, latent_num_rows, latent_num_columns, hidden_dim, action_dim, *, num_layers=1, device='cpu'):
        super().__init__()
        self.latent_dim = latent_num_columns * latent_num_rows
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device

        self.flatten = nn.Flatten(start_dim=2)
        self.GRU = nn.GRUCell(
            input_size=self.latent_dim + action_dim,
            hidden_size=hidden_dim,
            device=device
        )

    def forward(self, last_latent_state: torch.tensor, last_hidden_state: torch.tensor, last_action: torch.tensor):
        last_latent_state = self.flatten(last_latent_state)
        input_tensor = torch.cat((last_latent_state, last_action), dim=-1).squeeze(1)
        last_hidden_state = last_hidden_state.squeeze(1)
        hidden = self.GRU(input_tensor, last_hidden_state)
        return hidden.unsqueeze(1)

