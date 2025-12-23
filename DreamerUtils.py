import torch 
import numpy as np

def gaussian_log_probability(x: torch.tensor, mu: torch.tensor, sigma: torch.tensor) -> torch.tensor:
    """
    Calculate the log likelyhood of x apprearing in the gaussian distribution defined by mu and sigma.
    """
    dist = torch.distributions.Normal(loc=mu, scale=sigma)
    log_prob = dist.log_prob(x)
    return log_prob

def bernoulli_log_probability(p, k):
    epsilon = 1e-8
    p_clamped = torch.clamp(p, min=epsilon, max=1.0 - epsilon)
    log_prob = k * torch.log(p_clamped) + (1 - k) * torch.log(1 - p_clamped)
    return log_prob

def kullback_leibler_divergence_between_gaussians(
        mu_1: torch.tensor, 
        sigma_1: torch.tensor, 
        mu_2: torch.tensor, 
        sigma_2: torch.tensor
    ) -> torch.tensor:
    var_1 = torch.square(sigma_1) ; var_2 = torch.square(sigma_2)
    mean_diff = torch.square(mu_1 - mu_2)
    Dkl = torch.log(sigma_2 / sigma_1) + ((var_1 + mean_diff) / (2 * var_2)) - 0.5
    return Dkl

def symlog(x):
    return torch.sign(x) * torch.log(1.0 + torch.abs(x))

def symlog_np(x):
    return np.sign(x) * np.log(1.0 + np.abs(x))

def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

def to_twohot(value: torch.tensor, buckets: torch.tensor):
    clipped_value = torch.clamp(max=buckets.max(), min=buckets.min(), input=value)
    lower_bucket_idx = torch.searchsorted(buckets, clipped_value, right=True) - 1
    lower_bucket_val = buckets[lower_bucket_idx]
    upper_bucket_val = buckets[lower_bucket_idx + 1]
    weight = (clipped_value - lower_bucket_val) / (upper_bucket_val - lower_bucket_val + 1e-8)
    twohot_shape = value.shape[:-1] + (buckets.shape[0],)
    twohot = torch.zeros(twohot_shape, dtype=torch.float32, device=value.device)
    twohot = torch.scatter(twohot, dim=-1, index=lower_bucket_idx, src=(1.0 - weight))
    twohot = torch.scatter(twohot, dim=-1, index=(lower_bucket_idx + 1), src=weight)
    return twohot

def _sanitize_for_save(data_list):
    """Helper to recursively convert GPU tensors to CPU floats/arrays"""
    clean_data = []
    for item in data_list:
        if isinstance(item, torch.Tensor):
            clean_data.append(item.detach().cpu().item())
        elif isinstance(item, list):
            # Handle list of tensors (like WM_loss per epoch)
            clean_data.append([x.detach().cpu().item() if isinstance(x, torch.Tensor) else x for x in item])
        else:
            clean_data.append(item)
    return np.array(clean_data)