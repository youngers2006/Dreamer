import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, num_filters_1, num_filters_2, latent_size, device='cpu'):
        """
        Takes obseravtion (usually image) and maps it to a latent state representation through a CNN.
        """
        super().__init__()
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
            nn.Linear(in_features=num_filters_2 * 2 * 2, out_features=512, device=device),
            nn.SiLU(),
            nn.Linear(in_features=512, out_features=latent_size, device=self.device)
        )
    def forward(self, x):
        x = self.feature_extractor(x)
        return self.latent_mapper(x)

class Decoder(nn.Module):
    """
    Takes a latent state and maps it to the image it was created by.
    """
    def __init__(self, latent_dim, num_filters_1, num_filters_2, hidden_dim=512, device='cpu'):
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
            nn.ConvTranspose2d(in_channels=num_filters_1, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.upscaler(x)
        return self.image_builder(x)