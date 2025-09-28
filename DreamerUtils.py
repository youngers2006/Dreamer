import torch 
import math

def gaussian_log_probability(x: torch.tensor, mu: torch.tensor, sigma: torch.tensor) -> torch.tensor:
    """
    Calculate the log likelyhood of x apprearing in the gaussian distribution defined by mu and sigma.
    """
    log_prob = - torch.log(sigma) - (0.5) * torch.log(2 * math.pi) - (0.5) * torch.square((x - mu) / sigma)
    return log_prob

def bernoulli_log_probability(p, k):
    log_prob = k * torch.log(p) + (1 - k) * torch.log(1 - p)
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

def symexp(x):
    return torch.sign(x) * torch.exp(1.0 + torch.abs(x))