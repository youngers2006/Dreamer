import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    Encoder segment of VAE, maps observations to latent state representation.
    """
    def __init__(
            self,
            observation_dims,
            hidden_state_dim,
            latent_num_rows,
            latent_num_columns,
            num_filters_1,
            num_filters_2,
            hidden_layer_nodes,
            device='cpu'
        ):
        super().__init__()

        # Initialise dimensions
        self.latent_size = latent_num_rows * latent_num_columns
        self.latent_num_rows = latent_num_rows
        self.latent_num_columns = latent_num_columns
        self.final_height = observation_dims[0] // 16
        self.final_width = observation_dims[1] // 16

        # check that image size is suitable
        if self.final_height < 1 or self.final_width < 1:
            raise ValueError(f"Input image {observation_dims} is too small for 4 layers of downsampling.")

        # create convolutional net to analyse images
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

        # initialise flattened image dimensions
        flattened_feature_size = (num_filters_2 * 4) * self.final_height * self.final_width
        total_in_features = flattened_feature_size + hidden_state_dim

        # create network to map analysed images to latent states
        self.flatten = nn.Flatten(start_dim=2)
        self.latent_mapper = nn.Sequential(
            nn.Linear(in_features=total_in_features, out_features=hidden_layer_nodes, device=device),
            nn.LayerNorm(hidden_layer_nodes, device=device),
            nn.SiLU(),
            nn.Linear(in_features=hidden_layer_nodes, out_features=self.latent_size, device=device)
        )

    def forward(self, hidden, observation):
        # get initial shape
        B, S, C, H, W = observation.shape

        # resize observation to combine batch and sequence into a single dimension
        observation = observation.view(B * S, C, H, W)

        # extract features from observation using convnet
        features = self.feature_extractor(observation)

        # resize features to reseparate batch and sequence
        _, out_C, out_H, out_W = features.shape
        features = features.view(B, S, out_C, out_H, out_W)

        # flatten features and use the FF network to map to latent logits
        features = self.flatten(features)
        input = torch.cat((features, hidden), dim=-1)
        logits = self.latent_mapper(input)
        return logits
    
    def encode(self, hidden_state, observation):
        # get initial hidden state shape
        B, S, _ = hidden_state.shape

        # run forward class method to extract features and map the observation to latent logits
        logits = self.forward(hidden_state, observation)

        # logits are returned as a flat vector so resize them back to matrix
        logits = logits.view(B, S, self.latent_num_rows, self.latent_num_columns)

        # convert logits to probability with softmax
        probs = torch.softmax(logits.float(), dim=-1)

        # average probabilities with uniform distribution to prevent the distribution collapsing to deterministic
        uniform = (1.0 / self.latent_num_columns)
        probs = 0.99 * probs + 0.01 * uniform

        # sample from distribution to obtain a latent state, using sampling trick to allow gradient flow (STE)
        dist = torch.distributions.Categorical(probs=probs)
        sampled_idx = dist.sample()
        latent_state_OH = torch.nn.functional.one_hot(sampled_idx, num_classes=self.latent_num_columns).float()
        latent_state = latent_state_OH + probs - probs.detach()
        return latent_state, logits

class Decoder(nn.Module):
    """
    Decoder section of the VAE, maps latent states back to observations.
    """
    def __init__(self, latent_num_rows, latent_num_columns, observation_dim, hidden_state_dim, num_filters_1, num_filters_2, hidden_layer_nodes, device='cpu'):
        super().__init__()

        # initialise dimensions
        self.start_height = observation_dim[0] // 16
        self.start_width = observation_dim[1] // 16
        self.num_filters_start = num_filters_2 * 4

        self.hidden_dim = hidden_state_dim
        self.latent_row_dim = latent_num_rows
        self.latent_col_dim = latent_num_columns
        self.flatten = nn.Flatten(start_dim=1)

        # create network to map latent state back to observation features
        self.upscaler = nn.Sequential(
            nn.Linear(in_features=latent_num_rows * latent_num_columns + hidden_state_dim, out_features=hidden_layer_nodes, device=device),
            nn.LayerNorm(hidden_layer_nodes, device=device),
            nn.SiLU(),
            nn.Linear(in_features=hidden_layer_nodes, out_features=self.num_filters_start * self.start_height * self.start_width, device=device),
            nn.SiLU()
        )

        # create transposed convnet to reconstruct image from reconstructed features
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
        # resize latent and hidden states to combine batch and sequence dims
        B, S, _ = hidden.shape
        hidden = hidden.view(B * S, self.hidden_dim)
        latent = latent.view(B * S, self.latent_row_dim, self.latent_col_dim)

        # flatten latent state to allow feature reconstruction with linear net
        latent = self.flatten(latent)

        # concatenate the hidden and latent states for the upscaling net
        up_input = torch.cat((hidden, latent), dim=-1)
        x = self.upscaler(up_input)

        # change size of feature vector back from flat vector
        x = x.view(-1, self.num_filters_start, self.start_height, self.start_width)

        # reconstruct observation with transposed convnet
        mu = self.image_builder(x)

        # resize the observation to original shape
        _, C, H, W = mu.shape
        mu = mu.view(B, S, C, H, W)
        return mu
    
    def decode(self, hidden_state: torch.tensor, latent_state: torch.tensor):
        # return only the mean observation obtained from forward method to keep decoder deterministic
        mu = self.forward(hidden_state, latent_state)
        return mu