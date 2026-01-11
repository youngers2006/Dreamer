import torch
import torch.nn as nn

class Encoder(nn.Module): 
    def __init__(self, observation_dims, hidden_state_dim, latent_num_rows, latent_num_columns, num_filters_1, num_filters_2, hidden_layer_nodes, device='cpu'):
        """
        Takes obseravtion (image in this class) and maps it to a latent state representation through a CNN.
        """ 
        super().__init__()
        self.latent_size = latent_num_rows * latent_num_columns
        self.latent_num_rows = latent_num_rows
        self.latent_num_columns = latent_num_columns
        self.final_height = observation_dims[0] // 16
        self.final_width = observation_dims[1] // 16

        if self.final_height < 1 or self.final_width < 1:
            raise ValueError(f"Input image {observation_dims} is too small for 4 layers of downsampling.")

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, num_filters_1, kernel_size=4, stride=2, padding=1, device=device),
            nn.SiLU(),
            nn.Conv2d(num_filters_1, num_filters_2, kernel_size=4, stride=2, padding=1, device=device),
            nn.SiLU(),
            nn.Conv2d(num_filters_2, num_filters_2 * 2, kernel_size=4, stride=2, padding=1, device=device),
            nn.SiLU(),
            nn.Conv2d(num_filters_2 * 2, num_filters_2 * 4, kernel_size=4, stride=2, padding=1, device=device),
            nn.SiLU(),
        )
        flattened_feature_size = (num_filters_2 * 4) * self.final_height * self.final_width
        total_in_features = flattened_feature_size + hidden_state_dim
        self.flatten = nn.Flatten(start_dim=2)
        self.latent_mapper = nn.Sequential(
            nn.Linear(in_features=total_in_features, out_features=hidden_layer_nodes, device=device),
            nn.LayerNorm(hidden_layer_nodes, device=device),
            nn.SiLU(),
            nn.Linear(in_features=hidden_layer_nodes, out_features=self.latent_size, device=device)
        )

    def forward(self, hidden, observation):
        B, S, C, H, W = observation.shape
        observation = observation.view(B * S, C, H, W)
        features = self.feature_extractor(observation)
        _, out_C, out_H, out_W = features.shape
        features = features.view(B, S, out_C, out_H, out_W)
        features = self.flatten(features)
        input = torch.cat((features, hidden), dim=-1)
        logits = self.latent_mapper(input)
        return logits
    
    def encode(self, hidden_state, observation):
        B, S, _ = hidden_state.shape
        logits = self.forward(hidden_state, observation)
        logits = logits.view(B, S, self.latent_num_rows, self.latent_num_columns)
        probs = torch.softmax(logits, dim=-1)
        uniform = (1.0 / self.latent_num_columns)
        probs = 0.99 * probs + 0.01 * uniform
        dist = torch.distributions.Categorical(probs=probs)
        sampled_idx = dist.sample()
        latent_state_OH = torch.nn.functional.one_hot(sampled_idx, num_classes=self.latent_num_columns).float()
        latent_state = latent_state_OH + probs - probs.detach()
        return latent_state, logits

class Decoder(nn.Module):
    """
    Takes a latent state and maps it to the image it was created by.
    """
    def __init__(self, latent_num_rows, latent_num_columns, observation_dim, hidden_state_dim, num_filters_1, num_filters_2, hidden_layer_nodes, device='cpu'):
        super().__init__()
        self.start_height = observation_dim[0] // 16
        self.start_width = observation_dim[1] // 16
        self.num_filters_start = num_filters_2 * 4 

        self.hidden_dim = hidden_state_dim
        self.latent_row_dim = latent_num_rows
        self.latent_col_dim = latent_num_columns
        self.flatten = nn.Flatten(start_dim=1)

        self.upscaler = nn.Sequential(
            nn.Linear(in_features=latent_num_rows * latent_num_columns + hidden_state_dim, out_features=hidden_layer_nodes, device=device),
            nn.SiLU(),
            nn.Linear(in_features=hidden_layer_nodes, out_features=self.num_filters_start * self.start_height * self.start_width, device=device),
            nn.SiLU()
        )
        self.image_builder = nn.Sequential(
            nn.ConvTranspose2d(self.num_filters_start, num_filters_2 * 2, kernel_size=4, stride=2, padding=1, device=device),
            nn.SiLU(),
            nn.ConvTranspose2d(num_filters_2 * 2, num_filters_2, kernel_size=4, stride=2, padding=1, device=device),
            nn.SiLU(),
            nn.ConvTranspose2d(num_filters_2, num_filters_1, kernel_size=4, stride=2, padding=1, device=device),
            nn.SiLU(),
            nn.ConvTranspose2d(num_filters_1, 3, kernel_size=4, stride=2, padding=1, device=device),
            nn.Tanh()
        )

    def forward(self, hidden: torch.tensor, latent: torch.tensor):
        B, S, _ = hidden.shape
        hidden = hidden.view(B * S, self.hidden_dim)
        latent = latent.view(B * S, self.latent_row_dim, self.latent_col_dim)
        latent = self.flatten(latent)
        input = torch.cat((hidden, latent), dim=-1)
        x = self.upscaler(input)
        x = x.view(-1, self.num_filters_start, self.start_height, self.start_width)
        mu = self.image_builder(x)
        _, C, H, W = mu.shape
        mu = mu.view(B, S, C, H, W)
        return mu
    
    def decode(self, hidden_state: torch.tensor, latent_state: torch.tensor):
        mu = self.forward(hidden_state, latent_state)
        return mu