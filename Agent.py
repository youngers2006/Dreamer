import torch
import torch.nn as nn
from torch.distributions import Normal, TanhTransform, TransformedDistribution
from DreamerUtils import symexp, to_twohot, symlog
import copy

class Agent(nn.Module): # batched sequence (batch_size, sequence_length, features*)
    def __init__(
            self, 
            action_dim,
            latent_dims,
            hidden_state_dim,
            HL_A1,
            HL_A2,
            HL_C1,
            HL_C2, 
            critic_buckets,
            A_lr, 
            A_betas, 
            A_eps,
            C_lr, 
            C_betas, 
            C_eps,
            nu,
            lambda_,
            gamma,
            *, 
            device='cpu'
        ):
        super().__init__()
        self.device = device
        self.actor = Actor(
            action_dim, 
            latent_dims[0], 
            latent_dims[1], 
            hidden_state_dim, 
            HL_A1, 
            HL_A2, 
            device=device
        )
        self.critic = Critic(
            latent_dims[0], 
            latent_dims[1], 
            hidden_state_dim, 
            HL_C1, 
            HL_C2, 
            critic_buckets, 
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
            eps=A_eps
        )
        self.critic_optimiser = torch.optim.AdamW(
            params=self.critic.parameters(),
            lr=C_lr,
            betas=(C_betas[0], C_betas[1]),
            eps=C_eps
        )
    
    def update_S(self, lambda_returns: torch.tensor):
        flat_returns = lambda_returns.detach().flatten()
        per095 = torch.quantile(flat_returns, 0.95)
        per005 = torch.quantile(flat_returns, 0.05)

        range = torch.max(per095 - per005, torch.tensor(1.0, dtype=torch.float32, device=self.device))
        alpha = (1.0 - self.smoothing_factor)
        self.S = (1.0 - alpha) * self.S + alpha * range

    def soft_update_target(self, tau=0.02):
        with torch.no_grad():
            for p_current, p_target in zip(self.critic.parameters(), self.target_critic.parameters()):
                p_current.data.mul_(1.0 - tau)
                p_current.data.add_(tau * p_target.data)

    def train_step(self, z_batch_seq, h_batch_seq, reward_batch_seq, continue_batch_seq, action_batch_seq, a_mu_batch_seq, a_sigma_batch_seq):
        R_lambda_batch_seq = self.compute_batched_R_lambda_returns(
                h_batch_seq,
                z_batch_seq,
                reward_batch_seq,
                continue_batch_seq,
                continue_batch_seq.shape[1]
            )
        value_batched_seq = self.critic.value(h_batch_seq.detach(), z_batch_seq.detach())
        baseline = value_batched_seq[:, :-1]
        advantage_batched_seq = (R_lambda_batch_seq - baseline).detach()
        advantage_batched_seq = advantage_batched_seq.squeeze(-1)

        base_dist = Normal(loc=a_mu_batch_seq, scale=a_sigma_batch_seq)
        a_dist_batch_seq = TransformedDistribution(base_dist, [TanhTransform()])

        log_prob_batch_seq = a_dist_batch_seq.log_prob(action_batch_seq.detach()).sum(dim=-1)
        log_prob_batch_seq = log_prob_batch_seq[:, :-1]
        actor_entropy = -log_prob_batch_seq

        self.update_S(R_lambda_batch_seq)
        normalisation_term = torch.max(self.S, torch.tensor(1.0, dtype=torch.float32, device=self.device)).detach()
        scaled_advantage = (advantage_batched_seq / normalisation_term)

        loss_policy = - (log_prob_batch_seq * scaled_advantage)
        loss_entropy = - (self.nu * actor_entropy)
        loss_actor = torch.mean(loss_policy + loss_entropy)

        critic_logits = self.critic(h_batch_seq.detach(), z_batch_seq.detach())[:, :-1]

        target_returns = R_lambda_batch_seq.detach()
        target_returns_symlog = symlog(target_returns)
        R_lambda_th_batch_seq = to_twohot(target_returns_symlog, self.critic.buckets_crit)

        value_log_probs = nn.functional.log_softmax(critic_logits, dim=-1)
        loss_batched_seq_critic = -torch.sum(R_lambda_th_batch_seq * value_log_probs, dim=-1)
        loss_critic = torch.mean(loss_batched_seq_critic)

        if torch.isnan(loss_actor) or torch.isnan(loss_critic):
            print("Agent loss is NAN, skipping update.")
            return loss_actor, loss_critic

        self.critic_optimiser.zero_grad()
        loss_critic.backward(retain_graph=False)

        self.actor_optimiser.zero_grad()
        loss_actor.backward()

        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 50.0)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 50.0)

        self.critic_optimiser.step()
        self.actor_optimiser.step()

        self.soft_update_target()
        return loss_actor, loss_critic
    
    def compute_batched_R_lambda_returns(self, hidden_state_batched_seq, latent_state_batched_seq, reward_batched_seq, continue_batched_seq, seq_length):
        value_estimate_seq = self.target_critic.value(hidden_state_batched_seq, latent_state_batched_seq)
        next_return = value_estimate_seq[:, -1]
        R_lambda_seq = []
        for t in reversed(range(seq_length - 1)):
            reward_t = reward_batched_seq[:, t]
            continue_t = continue_batched_seq[:, t]
            value_t_plus_1 = value_estimate_seq[:, t+1]
            if reward_t.dim() == 1:
                reward_t = reward_t.unsqueeze(-1)
            if continue_t.dim() == 1:
                continue_t = continue_t.unsqueeze(-1)
            R_lambda = reward_t + self.gamma * continue_t * ((1 - self.lambda_) * value_t_plus_1 + self.lambda_ * next_return)
            R_lambda_seq.insert(0, R_lambda)
            next_return = R_lambda
        R_lambda_seq = torch.stack(R_lambda_seq, dim=1)
        return R_lambda_seq
        
class Actor(nn.Module):
    def __init__(self, action_dim, latent_column_dim, latent_row_dim, hidden_state_dim, hidden_layer_num_nodes_1, hidden_layer_num_nodes_2,*, device='cpu'):
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

    def forward(self, ht, zt):
        flattened_zt = self.flatten(zt)
        st = torch.cat([ht, flattened_zt], dim=-1)
        base_result = self.base_net(st)
        mu = self.mu_head(base_result)
        log_sig = self.log_sig_head(base_result)
        sigma = torch.nn.functional.softplus(log_sig) + 1e-4
        return mu, sigma
    
    def act(self, ht, zt, deterministic=False):
        mu, sigma = self.forward(ht, zt)
        if deterministic:
            action = torch.tanh(mu)
        else:
            base_dist = Normal(mu, sigma)
            dist = TransformedDistribution(base_dist, [TanhTransform()])
            action = dist.rsample()
        return action, mu, sigma

class Critic(nn.Module):
    def __init__(self, latent_row_dim, latent_column_dim, hidden_state_dim, hidden_layer_num_nodes_1, hidden_layer_num_nodes_2, num_buckets=255, device='cpu'):
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
        bucket_values = torch.linspace(-20, 20, num_buckets, device=device)
        self.register_buffer('buckets_crit', bucket_values)

    def forward(self, ht, zt):
        flattened_zt = self.flatten(zt)
        st = torch.cat([ht, flattened_zt], dim=-1)
        logits = self.value_net(st)
        return logits
    
    def value(self, ht, zt):
        logits = self.forward(ht, zt)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        symlog_value = torch.sum(probs * self.buckets_crit, dim=-1, keepdim=True)
        return symexp(symlog_value) 