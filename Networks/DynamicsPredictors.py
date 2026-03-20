import torch
import torch.nn as nn

# file module imports
from Utils import symexp

class DynamicsPredictor(nn.Module):
    """Next latent state predictor network section of the Recurrent State Space Model (RSSM).

    This class is a FF-network that outputs the latent state for the current timestep
    given the current hidden state.

    Args:
        latent_num_rows (int): Number of rows in the discrete latent state matrix.
        latent_num_columns (int): Number of columns in the discrete latent state matrix.
        hidden_state_size (int): Size of hidden state vector.
        hidden_L1 (int): Number of neurons in the first hidden state of the network.
        hidden_L2 (int): Number of neurons in the second hidden state of the network.
        device (str, optional): Storage location of the network ('cpu' or 'cuda'). 
            Defaults to 'cpu'.

    Attributes:
        latent_num_rows (int): Number of rows in the latent state matrix.
        latent_num_columns (int): Number of columns in the latent state matrix.
        latent_size (int): Size of flattened latent state matrix.
        device (str): Storage location of the network ('cpu' or 'cuda').
    """
    def __init__(
            self, latent_num_rows: int, latent_num_columns: int,
            hidden_state_size: int, hidden_L1: int, hidden_L2: int,
            *, device='cpu'
        ):
        super().__init__()
        self.latent_num_rows = latent_num_rows
        self.latent_num_columns = latent_num_columns
        self.latent_size = latent_num_rows * latent_num_columns
        self.device = device

        self.logit_net = nn.Sequential(
            nn.Linear(in_features=hidden_state_size, out_features=hidden_L1, device=device),
            nn.LayerNorm(hidden_L1, device=device),
            nn.SiLU(),
            nn.Linear(in_features=hidden_L1, out_features=hidden_L2, device=device),
            nn.LayerNorm(hidden_L2, device=device),
            nn.SiLU(),
            nn.Linear(in_features=hidden_L2, out_features=self.latent_size, device=device)
        )

    def forward(self, x: torch.Tensor):
        """Runs network forward pass to compute prediction logits.

        This function runs a forward pass of the dynamics networks to compute prediction
        logits for the input x.

        Args:
            x (torch.Tensor): Network input.

        Returns:
            logits (torch.Tensor): Prediction logits.
        """
        # Run forward pass to compute logits
        logits = self.logit_net(x)

        # resize logits tensor in feature dimensions to recover matrix and return logits
        batch_dim, seq_dim, _ = logits.shape
        logits = logits.view(
            batch_dim, seq_dim, self.latent_num_rows, self.latent_num_columns
        )
        return logits
   
    def predict(self, hidden_state: torch.Tensor):
        """Runs dynamics predictor to compute latent state from hidden state.

        This function computes the stochastic discrete latent state given the hidden state.

        Args:
            hidden_state (torch.Tensor): Current hidden state, h_t.

        Returns:
            latent_state: Current latent state, z_t.
            logits: Prediction logits of the network, used for training.
        """
        # Run forward pass to compute prediction logits
        logits = self.forward(hidden_state)

        # Compute probability distribution of latent states and sample from it
        probs = torch.softmax(logits, dim=-1)
        uniform = 1.0 / self.latent_num_columns
        probs = 0.99 * probs + 0.01 * uniform
        dist = torch.distributions.Categorical(probs=probs)
        sample_idx = dist.sample()

        # Construct latent state matrix with onehot encodings and use STE to allow gradient flow
        latent_state_one_hot = torch.nn.functional.one_hot(
            sample_idx, num_classes=self.latent_num_columns
        ).float()
        latent_state = latent_state_one_hot + probs - probs.detach()
        return latent_state, logits

class RewardPredictor(nn.Module):
    """Reward predictor network.

    This class is a FF-network that outputs the predicted reward for the current state.

    Args:
        latent_num_rows (int): Number of rows in the discrete latent state matrix.
        latent_num_columns (int): Number of columns in the discrete latent state matrix.
        hidden_state_size (int): Size of hidden state vector.
        hidden_L1 (int): Number of neurons in the first hidden state of the network.
        hidden_L2 (int): Number of neurons in the second hidden state of the network.
        num_buckets (int): Number of reward buckets available.
        device (str, optional): Storage location of the network ('cpu' or 'cuda'). 
            Defaults to 'cpu'.

    Attributes:
        latent_num_rows (int): Number of rows in the latent state matrix.
        latent_num_columns (int): Number of columns in the latent state matrix.
        latent_size (int): Size of flattened latent state matrix.
        device (str): Storage location of the network ('cpu' or 'cuda').
    """
    def __init__(
            self, latent_num_rows: int, latent_num_columns: int,
            hidden_state_size: int, hidden_L1: int, hidden_L2: int,
            num_buckets=255, device='cpu'
        ):
        super().__init__()
        self.latent_size = latent_num_rows * latent_num_columns
        self.buckets = num_buckets
        self.device = device

        self.flatten = nn.Flatten(start_dim=2)
        self.logit_net = nn.Sequential(
            nn.Linear(
                in_features=hidden_state_size + self.latent_size,
                out_features=hidden_L1, device=device
            ),
            nn.LayerNorm(hidden_L1, device=device),
            nn.SiLU(),
            nn.Linear(
                in_features=hidden_L1, out_features=hidden_L2,
                device=device
            ),
            nn.LayerNorm(hidden_L2, device=device),
            nn.SiLU(),
            nn.Linear(
                in_features=hidden_L2, out_features=num_buckets,
                device=device
            )
        )

        # Setup reward buckets to allow
        buckets = torch.linspace(
            -20.0, 20.0, num_buckets, device=device
        )
        self.register_buffer('buckets_rew', buckets)

    def forward(self, hidden: torch.Tensor, latent: torch.Tensor):
        """Runs network forward pass to compute prediction logits.

        This function runs a forward pass of the reward network, computes reward logits
        for each reward bucket.

        Args:
            hidden (torch.Tensor): Current hidden state, h_t.
            latent (torch.Tensor): Current latent state, z_t.

        Returns:
            logits (torch.Tensor): Prediction logits.
        """

        # Concatenate hidden and flattned latent states along the feature dims
        latent = self.flatten(latent)
        input = torch.cat([hidden, latent], dim=-1)

        # Compute reward logits
        logits = self.logit_net(input)
        return logits

    def predict(self, hidden_state: torch.Tensor, latent_state: torch.Tensor):
        """Runs reward predictor to compute latent state from hidden state.

        This function computes the stochastic discrete latent state given the hidden state.

        Args:
            hidden_state (torch.Tensor): Current hidden state, h_t.
            latent_state (torch.Tensor): Current latent state, z_t.

        Returns:
            reward (torch.Tensor): Predicted reward buckets.
        """

        # Compute symlog reward buckets
        logits = self.forward(hidden_state, latent_state)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        symlog_reward = torch.sum(probs * self.buckets_rew, dim=-1, keepdim=True)

        # Convert symlog rewards back to reward buckets and return
        reward = symexp(symlog_reward)
        return reward

class ContinuePredictor(nn.Module):
    """Reward predictor network.

    This class is a FF-network that outputs the predicted reward for the current state.

    Args:
        latent_num_rows (int): Number of rows in the discrete latent state matrix.
        latent_num_columns (int): Number of columns in the discrete latent state matrix.
        hidden_state_size (int): Size of hidden state vector.
        hidden_L1 (int): Number of neurons in the first hidden state of the network.
        hidden_L2 (int): Number of neurons in the second hidden state of the network.
        device (str, optional): Storage location of the network ('cpu' or 'cuda'). 
            Defaults to 'cpu'.

    Attributes:
        latent_size (int): Size of flattened latent state matrix.
        device (str): Storage location of the network ('cpu' or 'cuda').
    """
    def __init__(
            self, latent_num_rows: int, latent_num_columns: int,
            hidden_state_size: int, hidden_L1: int, hidden_L2: int,
            *, device='cpu'
        ):
        super().__init__()
        self.latent_size = latent_num_rows * latent_num_columns
        self.device = device

        self.flatten = nn.Flatten(start_dim=2)
        self.logit_generator = nn.Sequential(
            nn.Linear(
                in_features=hidden_state_size + self.latent_size, 
                out_features=hidden_L1, device=device
            ),
            nn.LayerNorm(hidden_L1, device=device),
            nn.SiLU(),
            nn.Linear(
                in_features=hidden_L1, out_features=hidden_L2,
                device=device
            ),
            nn.LayerNorm(hidden_L2, device=device),
            nn.SiLU(),
            nn.Linear(
                in_features=hidden_L2, out_features=1, device=device
            )
        )

    def forward(self, hidden: torch.Tensor, latent: torch.Tensor):
        """Runs network forward pass to compute continue probability.

        This function runs a forward pass of the continue network, 
        computes continue probability.

        Args:
            hidden (torch.Tensor): Current hidden state, h_t.
            latent (torch.Tensor): Current latent state, z_t.

        Returns:
            probability (torch.Tensor): Probability of continuing.
            logits (torch.Tensor): Prediction logits, for training.
        """

        # concat flattened latent state and hidden state along feature dim
        latent = self.flatten(latent)
        net_input = torch.cat([hidden, latent], dim=-1)

        # Computes probability of continuation
        logit = self.logit_generator(net_input)
        probability = torch.sigmoid(logit)
        return probability, logit

    def predict(self, hidden_state: torch.Tensor, latent_state: torch.Tensor):
        """Computes continue probability.

        This function computes continue probability.

        Args:
            hidden_state (torch.Tensor): Current hidden state, h_t.
            latent_state (torch.Tensor): Current latent state, z_t.

        Returns:
            probability (torch.Tensor): Probability of continuing, continue_ = (probability >= 0.5).
        """

        # Run forward pass and return output
        probability, _ = self.forward(hidden_state, latent_state)
        return probability