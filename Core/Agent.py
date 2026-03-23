import torch
import torch.nn as nn
from torch.distributions import Normal, TanhTransform, TransformedDistribution
import copy

# file module imports
from Utils import symexp, to_twohot, symlog

class Agent(nn.Module): # batched sequence (batch_size, sequence_length, features*)
    """Agent class.

    This class is the Agent class, it consists of an actor and a critic. Training
    logic is self contained and uses REINFORCE. To allow for the use of a 
    discrete latent state, gradients are not propagated through the dynamics.

    Args:
        action_dim (int): Size of the action vector.
        latent_dims (tuple(int, int)): Dimensions of the latent state matrix.
        hidden_state_dim (int): Size of the hidden state vector.
        HL_A1 (int): Number of neurons in actor hidden layer 1.
        HL_A2 (int): Number of neurons in actor hidden layer 2.
        HL_C1 (int): Number of neurons in critic hidden layer 1.
        HL_C2 (int): Number of neurons in critic hidden layer 2.
        critic_buckets (int): Number of critic buckets.
        A_lr (float): Actor learn rate.
        A_betas (tuple(float, float)): Adam optimiser hyperparams for actor.
        A_eps (float): Adam optimiser hyperparams for actor.
        C_lr (float): Critic learn rate.
        C_betas (tuple(float, float)): Adam optimiser hyperparams for critic.
        C_eps (float): Adam optimiser hyperparams for critic.
        nu (float): Entropy regulariser coefficient, scales reward for high entropy in 
        action distribution to encourage exploration.
        lambda_ (float): Trace decay parameter, balances critic bootstrapping (lambda=0) 
        and monte carlo estimation (lambda=1) for returns, critic bootstrapping has high bias 
        but low variance, monte carlo returns have high variance.
        gamma (float): Discount factor to degrade future rewards.
        device (str, optional): Storage location of the network ('cpu' or 'cuda'). 
            Defaults to 'cpu'.

    Attributes:
        nu (float): Entropy regulariser coefficient, scales reward for high entropy in 
        action distribution to encourage exploration.
        lambda_ (float): Trace decay parameter, balances critic bootstrapping (lambda=0) 
        and monte carlo estimation (lambda=1) for returns, critic bootstrapping has high bias 
        but low variance, monte carlo returns have high variance.
        gamma (float): Discount factor to degrade future rewards.
        buckets (int): Number of critic buckets.
        S (float): Advantage scaling factor, normalises advantage in gradient update to handle
        diverse reward scaling across domains.
        smoothing_factor (float): Governs the EMA aggressiveness for updating S.
        device (str): Storage location of the network ('cpu' or 'cuda').
    """
    def __init__(
            self, action_dim: int, latent_dims: int,
            hidden_state_dim: int, HL_A1: int, HL_A2: int,
            HL_C1: int, HL_C2: int, critic_buckets: int,
            A_lr: float, A_betas: tuple[float, float], A_eps : float,
            C_lr: float, C_betas: tuple[float, float], C_eps: float,
            nu: float, lambda_: float, gamma: float,
            *,
            device='cpu'
        ):
        super().__init__()
        self.device = device
        self.actor = Actor(
            action_dim, latent_dims[0], latent_dims[1],
            hidden_state_dim, HL_A1, HL_A2,
            device=device
        )
        self.critic = Critic(
            latent_dims[0], latent_dims[1], hidden_state_dim,
            HL_C1, HL_C2, critic_buckets,
            device=device
        )
        self.target_critic = copy.deepcopy(self.critic)

        for p in self.target_critic.parameters():
            p.requires_grad = False

        self.nu = nu
        self.lambda_ = lambda_
        self.gamma = gamma
        self.buckets = critic_buckets

        self.S = 1.0
        self.smoothing_factor = 0.99

        self.actor_optimiser = torch.optim.AdamW(
            params=self.actor.parameters(),
            lr=A_lr,
            betas=(A_betas[0], A_betas[1]),
            eps=A_eps,
            weight_decay=1e-6
        )
        self.critic_optimiser = torch.optim.AdamW(
            params=self.critic.parameters(),
            lr=C_lr,
            betas=(C_betas[0], C_betas[1]),
            eps=C_eps,
            weight_decay=1e-6
        )

    def update_S(self, lambda_returns: torch.Tensor):
        """Updates advantage scaling factor.
        
        An exponential moving average to used to update the advantage scaling factor
        for stability, by using an advantage scaling factor the advantage always remains
        roughly normalised allowing use on different reward scales.

        Args:
            lambda_return (torch.Tensor): Tensor of trace decaying return values.
        """
        # Dont update if returns have crashed
        if torch.isnan(lambda_returns).any() or torch.isinf(lambda_returns).any():
            return

        # Calculate 5 and 95 percentile values to exclude outliers to get range of the returns
        flat_returns = lambda_returns.detach().flatten()
        per095 = torch.quantile(flat_returns, 0.95)
        per005 = torch.quantile(flat_returns, 0.05)
        range_val = torch.max(
            per095 - per005, torch.tensor(1.0, dtype=torch.float32, device=self.device)
        )

        # Update S with moving average
        alpha = 1.0 - self.smoothing_factor
        self.S = (1.0 - alpha) * self.S + alpha * range_val

    def soft_update_target(self, tau=0.02):
        """Performs soft update on target critic network parameters.

        Args:
            tau (float): Update size.
        """
        # Hold gradient tracking
        with torch.no_grad():

            # EMA to update target network to become current network
            for p_current, p_target in zip(
                self.critic.parameters(), self.target_critic.parameters()
            ):
                p_target.data.mul_(1.0 - tau)
                p_target.data.add_(tau * p_current.data)

    def compute_batched_R_lambda_returns(
            self, hidden_state_batched_seq: torch.Tensor, 
            latent_state_batched_seq: torch.Tensor, reward_batched_seq: torch.Tensor, 
            continue_batched_seq: torch.Tensor, seq_length: int
        ):
        """Computes trace returns for all steps in a batch of sequences.

        The returns are computed with a EMA between high variance monte carlo returns and 
        high bias critic bootstrapped estimation, the weighting of each component to 
        the returns governs the bias, variance tradeoff.

        Args:
            hidden_state_batched_seq (torch.Tensor): Batch of hidden state sequences.
            latent_state_batched_seq (torch.Tensor): Batch of latent state sequences.
            reward_batched_seq (torch.Tensor): Batch of reward sequences.
            continue_batched_seq (torch.Tensor): Batch of continue flag sequences.
            seq_length (int): Length of input sequences.

        Returns:
            R_lambda_seq (torch.Tensor): Trace returns for each 
        """
        # Get baseline value estimate from critic and compute 1 step boot strapped returns
        value_estimate_seq = self.target_critic.value(
            hidden_state_batched_seq, latent_state_batched_seq
        )

        # Calculate final monte carlo return step to begin
        next_return = reward_batched_seq[:, -1] + self.gamma * continue_batched_seq[:, -1] * value_estimate_seq[:, -1]
        R_lambda_seq = [next_return]

        # Loop back through time to calculate monte carlo returns
        for t in reversed(range(seq_length - 1)):
            reward_t = reward_batched_seq[:, t]
            continue_t = continue_batched_seq[:, t]
            value_t_plus_1 = value_estimate_seq[:, t + 1]

            # Fix dimenionality to add back feature dim
            if reward_t.dim() == 1:
                reward_t = reward_t.unsqueeze(-1)
            if continue_t.dim() == 1:
                continue_t = continue_t.unsqueeze(-1)

            # Use weighted average of critic prediction and future reward to get return
            R_lambda = reward_t + self.gamma * continue_t * ((1 - self.lambda_) * value_t_plus_1 + self.lambda_ * next_return)
            R_lambda_seq.insert(0, R_lambda)
            next_return = R_lambda

        # Stack all time elements back into sequences and return
        R_lambda_seq = torch.stack(R_lambda_seq, dim=1)
        return R_lambda_seq

    def train_step(
            self, z_batch_seq: torch.Tensor, h_batch_seq: torch.Tensor,
            reward_batch_seq: torch.Tensor, continue_batch_seq: torch.Tensor,
            action_batch_seq: torch.Tensor,
            a_mu_batch_seq: torch.Tensor, a_sigma_batch_seq: torch.Tensor
        ):
        """Computes trace returns for all steps in a batch of sequences.

        The returns are computed with a EMA between high variance monte carlo returns and 
        high bias critic bootstrapped estimation, the weighting of each component to 
        the returns governs the bias, variance tradeoff.

        Args:
            hidden_state_batched_seq (torch.Tensor): Batch of hidden state sequences.
            latent_state_batched_seq (torch.Tensor): Batch of latent state sequences.
            reward_batched_seq (torch.Tensor): Batch of reward sequences.
            continue_batched_seq (torch.Tensor): Batch of continue flag sequences.
            seq_length (int): Length of input sequences.

        Returns:
            R_lambda_seq (torch.Tensor): Trace returns for each 
        """
        # Computes returns from sequences
        R_lambda_batch_seq = self.compute_batched_R_lambda_returns(
                h_batch_seq,
                z_batch_seq,
                reward_batch_seq,
                continue_batch_seq,
                continue_batch_seq.shape[1]
        )

        # Compute baseline value with critic (expected return from state)
        value_batched_seq = self.critic.value(h_batch_seq.detach(), z_batch_seq.detach())
        baseline = value_batched_seq[:, :-1]

        # Compute advantage of each action compared to expectation
        # Detach to make sure gradients dont flow to the critic
        advantage_batched_seq = (R_lambda_batch_seq - baseline).detach()
        advantage_batched_seq = advantage_batched_seq.squeeze(-1)

        # Set up a tanh action distribution to reconstruct actor sampling distribution
        base_dist = Normal(loc=a_mu_batch_seq, scale=a_sigma_batch_seq)
        a_dist_batch_seq = TransformedDistribution(base_dist, [TanhTransform()])

        # Clamp just before 1 to prevent singularities in log prob and calculate the 
        # log probability  of each action from the action distribution
        action_batch_seq_clamped = torch.clamp(action_batch_seq.detach(), -1.0 + 1e-6, 1.0 - 1e-6)
        log_prob_batch_seq = a_dist_batch_seq.log_prob(action_batch_seq_clamped).sum(dim=-1)

        # Monte carlo estimate for actor entropy
        actor_entropy = -log_prob_batch_seq

        # Update adavnateg scaling term and use it to normalise the advantage
        self.update_S(R_lambda_batch_seq)
        normalisation_term = torch.max(self.S, torch.tensor(1.0, dtype=torch.float32, device=self.device)).detach()
        scaled_advantage = (advantage_batched_seq / normalisation_term)

        # Compute the policy loss and entropy loss for the actor
        loss_policy = - (log_prob_batch_seq * scaled_advantage)
        loss_entropy = - (self.nu * actor_entropy)
        loss_actor = torch.mean(loss_policy + loss_entropy)

        # Compute bucket prediction logits, detach sequences to stop BP through WM
        critic_logits = self.critic(h_batch_seq.detach(), z_batch_seq.detach())[:, :-1]

        # Convert rewards to symlog scale and encode into twohot
        target_returns = R_lambda_batch_seq.detach()
        target_returns_symlog = symlog(target_returns)
        R_lambda_th_batch_seq = to_twohot(target_returns_symlog, self.critic.buckets_crit)

        # Compute cross entropy loss for the critic bucket preictions
        value_log_probs = nn.functional.log_softmax(critic_logits, dim=-1)
        loss_batched_seq_critic = -torch.sum(R_lambda_th_batch_seq * value_log_probs, dim=-1)
        loss_critic = torch.mean(loss_batched_seq_critic)

        # Check all losses are valid before update step
        if torch.isnan(loss_actor) or torch.isinf(loss_actor) or torch.isnan(loss_critic) or torch.isinf(loss_critic):
            print("Agent loss is nan or inf, skipping update.")
            return loss_actor, loss_critic

        # Backpropagate Critic loss
        self.critic_optimiser.zero_grad()
        loss_critic.backward(retain_graph=False)

        # Backpropagate Actor loss
        self.actor_optimiser.zero_grad()
        loss_actor.backward()

        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 100.0)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0)

        # Update network weights
        self.critic_optimiser.step()
        self.actor_optimiser.step()

        # Step target Critic network weights toward current weights
        self.soft_update_target()
        return loss_actor, loss_critic

class Actor(nn.Module):
    """Actor class.

    This class is the Actor class. It is the policy the agents acts on. 
    It consists of a FF neural network trained to output a action distribution 
    that is sampled from to select actions.

    Args:
        action_dim (int): Size of the action vector.
        latent_column_dim (int): .
        latent_row_dim (int): .
        hidden_state_dim (int): Size of the hidden state vector.
        hidden_layer_num_nodes_1 (int): Number of neurons in hidden layer 1.
        hidden_layer_num_nodes_2 (int): Number of neurons in hidden layer 2.
        device (str, optional): Storage location of the network ('cpu' or 'cuda'). 
            Defaults to 'cpu'.
    """
    def __init__(
            self, action_dim: int, latent_column_dim: int, latent_row_dim: int, 
            hidden_state_dim: int, hidden_layer_num_nodes_1: int, hidden_layer_num_nodes_2: int,
            *,
            device='cpu'
        ):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=2)
        self.base_net = nn.Sequential(
            nn.Linear(in_features=latent_row_dim * latent_column_dim + hidden_state_dim, out_features=hidden_layer_num_nodes_1, device=device),
            nn.LayerNorm(hidden_layer_num_nodes_1, device=device),
            nn.SiLU(),
            nn.Linear(in_features=hidden_layer_num_nodes_1, out_features=hidden_layer_num_nodes_2, device=device),
            nn.LayerNorm(hidden_layer_num_nodes_2, device=device),
            nn.SiLU()
        )
        self.mu_head = nn.Linear(in_features=hidden_layer_num_nodes_2, out_features=action_dim, device=device)
        self.log_sig_head = nn.Linear(in_features=hidden_layer_num_nodes_2, out_features=action_dim, device=device)
        torch.nn.init.zeros_(self.mu_head.weight)
        torch.nn.init.zeros_(self.mu_head.bias)

    def forward(self, ht: torch.Tensor, zt: torch.Tensor):
        """Actor forward method.

        Computes action probability distribution given the current hidden state
        and latent state.

        Args:
            ht (torch.Tensor): Current hidden state.
            zt (torch.Tensor): Current latent state.

        Returns:
            mu (torch.Tensor): Mean of action distribution.
            sigma (torch.Tensor): Std.dev of action distribution.
        """
        flattened_zt = self.flatten(zt)
        st = torch.cat([ht, flattened_zt], dim=-1)
        base_result = self.base_net(st)

        mu = self.mu_head(base_result)
        log_sig = self.log_sig_head(base_result)
        log_sig = torch.clamp(log_sig, -5.0, 2.0)
        sigma = torch.nn.functional.softplus(log_sig) + 1e-3
        return mu, sigma
    
    def act(self, ht: torch.Tensor, zt: torch.Tensor, deterministic=False):
        """Actor act method.

        Computes action and action probability distribution given the current hidden state
        and latent state. All output actions are tanh transformed so sit between -1 and 1.

        Args:
            ht (torch.Tensor): Current hidden state.
            zt (torch.Tensor): Current latent state.
            deterministic (bool, optional): If the agent selects the most probable action (greedy),
                or samples from the distribution to explore. Defaults to False.

        Returns:
            action (torch.Tensor): Selected action from distribution
            mu (torch.Tensor): Mean of action distribution.
            sigma (torch.Tensor): Std.dev of action distribution.
        """
        mu, sigma = self.forward(ht, zt)
        if deterministic:
            action = torch.tanh(mu)
        else:
            # Create tanh transformed distribution as actions sit between -1 and 1
            base_dist = Normal(mu, sigma)
            dist = TransformedDistribution(base_dist, [TanhTransform()])

            # Use rsample to allow BP, as rsample uses STE
            action = dist.rsample()
        return action, mu, sigma

class Critic(nn.Module):
    """Critic class.

    This class is the Critic class, It is trained to approximate the value function 
    at any state, returning the expected value of any state following the current policy. 
    It consists of a FF neural network trained to output a action distribution 
    that is sampled from to select actions.

    Args:
        latent_row_dim (int): Number of rows in the latent state matrix.
        latent_column_dim (int): Number of columns in the latent state matrix.
        hidden_state_dim (int): Size of the hidden state vector.
        hidden_layer_num_nodes_1 (int): Number of neurons in hidden layer 1.
        hidden_layer_num_nodes_2 (int): Number of neurons in hidden layer 2.
        num_buckets (int, optional): Number of buckets to have twohot.
        value prediciton encoded into. Defaults to 255.
        device (str, optional): Storage location of the network ('cpu' or 'cuda'). 
            Defaults to 'cpu'.
    
    Attributes:
        latent_row_dim (int): Number of rows in the latent state matrix.
        latent_column_dim (int): Number of columns in the latent state matrix.
        num_buckets (int, optional): Number of buckets to have twohot.
    """
    def __init__(
            self, latent_row_dim: int, latent_column_dim: int,
            hidden_state_dim: int, hidden_layer_num_nodes_1: int,
            hidden_layer_num_nodes_2: int, num_buckets=255,
            device='cpu'
        ):
        super().__init__()
        self.latent_row_dim = latent_row_dim
        self.latent_column_dim = latent_column_dim
        self.num_buckets = num_buckets
        self.flatten = nn.Flatten(start_dim=2)
        self.value_net = nn.Sequential(
            nn.Linear(in_features=latent_column_dim * latent_row_dim + hidden_state_dim, out_features=hidden_layer_num_nodes_1, device=device),
            nn.LayerNorm(hidden_layer_num_nodes_1, device=device),
            nn.SiLU(),
            nn.Linear(in_features=hidden_layer_num_nodes_1, out_features=hidden_layer_num_nodes_2, device=device),
            nn.LayerNorm(hidden_layer_num_nodes_2, device=device),
            nn.SiLU(),
            nn.Linear(in_features=hidden_layer_num_nodes_2, out_features=num_buckets, device=device)
        )

        # Despie outputting a single scalar value the prediction process using buckets
        # improves satbility during training by allowing the use of cross entropy loss
        bucket_values = torch.linspace(-20, 20, num_buckets, device=device)
        self.register_buffer('buckets_crit', bucket_values)

    def forward(self, ht: torch.Tensor, zt: torch.Tensor):
        """Critic forward method.

        Computes value twohot logits given the current hidden state
        and latent state.

        Args:
            ht (torch.Tensor): Current hidden state.
            zt (torch.Tensor): Current latent state.

        Returns:
            logits (torch.Tensor): Value representing relative chance.
        """
        flattened_zt = self.flatten(zt)
        st = torch.cat([ht, flattened_zt], dim=-1)
        logits = self.value_net(st)
        return logits
    
    def value(self, ht: torch.Tensor, zt: torch.Tensor):
        """Actor forward method.

        Computes action probability distribution given the current hidden state
        and latent state.

        Args:
            ht (torch.Tensor): Current hidden state.
            zt (torch.Tensor): Current latent state.

        Returns:
            value (torch.Tensor): Value estimate of the current state.
        """
        logits = self.forward(ht, zt)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        symlog_value = torch.sum(probs * self.buckets_crit, dim=-1, keepdim=True)
        return symexp(symlog_value)