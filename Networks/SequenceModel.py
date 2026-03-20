import torch
import torch.nn as nn

class SequenceModel(nn.Module):
    """Sequence GRU network section of the Recurrent State Space Model (RSSM).

    This class is a GRU network that outputs the hidden state for the next timestep
    given previous hidden state, previous latent state and previous action taken.

    Args:
        latent_num_rows (int): Number of rows in the discrete latent state matrix.
        latent_num_columns (int): Number of columns in the discrete latent state matrix.
        hidden_dim (int): Size of the GRU hidden state vector.
        action_dim (int): Size of the action vector.
        device (str, optional): Storage location of the network ('cpu' or 'cuda'). 
            Defaults to 'cpu'.

    Attributes:
        latent_dim (int): Size of flattened discrete latent state matrix.
        hidden_dim (int): Size of hidden state.
        num_layers (int): Number of GRU layers.
        device (str): Storage location of the network ('cpu' or 'cuda').
    """
    def __init__(
            self, latent_num_rows: int, latent_num_columns: int,
            hidden_dim: int, action_dim: int, *, num_layers=1, device='cpu'
        ):
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

    def forward(
            self, last_latent_state: torch.Tensor,
            last_hidden_state: torch.Tensor, last_action: torch.Tensor
        ):
        """Runs network forward pass to compute next hidden state.

        This function computes the deterministic hidden state for the next timestep
        given previous hidden state, previous latent state and previous action taken.

        Args:
            last_latent_state (torch.Tensor): Previous latent state, z_t-1.
            last_hidden_state (torch.Tensor): Previous hidden state, h_t-1.
            last_action (torch.Tensor): Previous action, a_t-1.

        Returns:
            hidden: Next timestep hidden state, h_t.
        """
        # Flatten latent matrix into vector in seq and feature dimensions
        last_latent_state = self.flatten(last_latent_state)

        # Concatenate latent state and action along feature dimension and remove sequence dim
        input_tensor = torch.cat((last_latent_state, last_action), dim=-1).squeeze(1)
        last_hidden_state = last_hidden_state.squeeze(1)

        # Compute hidden state and add sequence dimension
        hidden = self.GRU(input_tensor, last_hidden_state)
        return hidden.unsqueeze(1)