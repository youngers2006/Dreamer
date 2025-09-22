import torch
import torch.nn as nn

class GatedRecurrentUnit(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, *, num_layers=1, device='cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.device = device

        self.GRU = nn.GRU(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=False,
            device=device
        )
        self.predictor = nn.Linear(
            in_features=hidden_dim, 
            out_features=output_dim, 
            device=device
        )
    def forward(self, x: torch.tensor, hidden: torch.tensor):
        gru_out, hidden_new = self.GRU(x, hidden)
        return self.predictor(gru_out.squeeze(0)), hidden_new

