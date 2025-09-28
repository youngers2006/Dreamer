import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_num_rows, latent_num_columns, num_filters_1, num_filters_2, hidden_layer_nodes, device='cpu'):
        """
        Takes obseravtion (image in this class) and maps it to a latent state representation through a CNN.
        """ 
        super().__init__()
        self.latent_size = latent_num_rows * latent_num_columns
        self.latent_num_rows = latent_num_rows
        self.latent_num_columns = latent_num_columns
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=num_filters_1, kernel_size=3, stride=1, padding=1, device=device),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=num_filters_1, out_channels=num_filters_2, kernel_size=3, stride=1, padding=1, device=device),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.latent_mapper = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=num_filters_2 * 2 * 2, out_features=hidden_layer_nodes, device=device),
            nn.SiLU(),
            nn.Linear(in_features=hidden_layer_nodes, out_features=self.latent_size, device=device)
        )

    def forward(self, x1, x2):
        features = self.feature_extractor(x2)
        _, _, H, W = features.shape
        x1 = x1.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)
        input = torch.cat((features, x1), axis=1)
        logits = self.latent_mapper(input)
        return logits
    
    def encode(self, hidden_state, observation):
        logits = self.forward(hidden_state, observation)
        dist = torch.distributions.Categorical(logits=logits)
        sampled_idx = dist.sample()
        latent_state_flat = torch.nn.functional.one_hot(sampled_idx, num_classes=self.latent_size)
        latent_state = latent_state_flat.view(-1, self.latent_num_rows, self.latent_num_columns, dtype=torch.float32)
        return latent_state, logits

class Decoder(nn.Module):
    """
    Takes a latent state and maps it to the image it was created by.
    """
    def __init__(self, latent_num_rows, latent_num_columns, observation_dim, hidden_state_dim, num_filters_1, num_filters_2, hidden_layer_nodes, device='cpu'):
        super().__init__()
        self.upscale_starting_dim = observation_dim / 4
        self.num_filters_2 = num_filters_2
        self.flatten = nn.Flatten()
        self.upscaler = nn.Sequential(
            nn.Linear(in_features=latent_num_rows * latent_num_columns + hidden_state_dim, out_features=hidden_layer_nodes),
            nn.SiLU(),
            nn.Linear(in_features=hidden_layer_nodes, out_features=num_filters_2 * self.upscale_starting_dim * self.upscale_starting_dim),
            nn.SiLU()
        )
        self.image_builder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=num_filters_2, out_channels=num_filters_1, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(in_channels=num_filters_1, out_channels=6, kernel_size=4, stride=2, padding=1)
        )
        self.softplus = nn.Softplus()

    def forward(self, x1: torch.tensor, x2: torch.tensor):
        x2 = self.flatten(x2)
        x = torch.cat((x1, x2), dim=-1)
        x = self.upscaler(x)
        x = x.view(-1, self.num_filters_2, self.upscale_starting_dim, self.upscale_starting_dim)
        obs_params = self.image_builder(x)
        mu, sigma_logits = torch.chunk(obs_params, chunks=2, dim=1)
        sigma = self.softplus(sigma_logits) + 1e-4
        return mu, sigma
    
    def decode(self, hidden_state: torch.tensor, latent_state: torch.tensor):
        mu, sigma = self.forward(hidden_state, latent_state)
        dist = torch.distributions.Independent(torch.distributions.Normal(loc=mu, scale=sigma), 3)
        observation = dist.rsample()
        return observation, mu, sigma


    