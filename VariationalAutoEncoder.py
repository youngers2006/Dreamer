import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, num_filters_1, num_filters_2, latent_size, device='cpu'):
        """
        Takes obseravtion (image in this class) and maps it to a latent state representation through a CNN.
        """
        super().__init__()
        self.latent_size = latent_size
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=num_filters_1, kernel_size=3, stride=1, padding=1, device=device),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=num_filters_1, out_channels=num_filters_2, kernel_size=3, stride=1, padding=1, device=device),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.latent_mapper_base = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=num_filters_2 * 2 * 2, out_features=512, device=device),
            nn.SiLU(),
        )
        self.mu_head = nn.Linear(in_features=512, out_features=latent_size, device=device)
        self.log_sig_head = nn.Linear(in_features=512, out_features=latent_size, device=device)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.latent_mapper_base(x)
        mu = self.mu_head(x)
        log_sig = self.log_sig_head(x)
        sigma = torch.exp(log_sig)
        return mu, sigma
    
    def encode(self, observation):
        mu, sigma = self.forward(observation)
        dist = torch.distributions.Normal(loc=mu, scale=sigma)
        latent_state = dist.rsample()
        return latent_state

class Decoder(nn.Module):
    """
    Takes a latent state and maps it to the image it was created by.
    """
    def __init__(self, latent_dim, observation_dims, num_filters_1, num_filters_2, hidden_dim=512, device='cpu'):
        super().__init__()
        self.upscaler = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=hidden_dim),
            nn.SiLU(),
            nn.Linear(in_features=hidden_dim, out_features=num_filters_2 * 8 * 8),
            nn.SiLU()
        )
        self.image_builder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=num_filters_2, out_channels=num_filters_1, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(in_channels=num_filters_1, out_channels=6, kernel_size=4, stride=2, padding=1)
        )
        self.softplus = nn.Softplus()

    def forward(self, x):
        obs_params = self.upscaler(x)
        mu, sigma_logits = torch.chunk(obs_params, chunks=2, dim=1)
        sigma = self.softplus(sigma_logits) + 1e-4
        return mu, sigma
    
    def decode(self, latent_state):
        mu, sigma = self.forward(latent_state)
        dist = torch.distributions.Independent(torch.distributions.Normal(loc=mu, scale=sigma), 3)
        observation = dist.rsample()


    