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

        # Ensure all inputs are at least 2D for concatenation
        if last_latent_state.dim() == 1:
            last_latent_state = last_latent_state.unsqueeze(0)
        if last_action.dim() == 1:
            last_action = last_action.unsqueeze(0)

        # The hidden state for a GRU must be 3D: (num_layers, batch_size, hidden_size)
        if last_hidden_state.dim() == 2:
            last_hidden_state = last_hidden_state.unsqueeze(0)

        input_tensor = torch.cat((last_latent_state, last_action), dim=-1)
        
        # The input to a GRU must be 3D: (sequence_length, batch_size, input_size)
        # Here, we are processing a single time step, so sequence_length is 1.
        if input_tensor.dim() == 2:
            input_tensor = input_tensor.unsqueeze(0)

        _, hidden = self.GRU(input_tensor, last_hidden_state)
        return hidden

