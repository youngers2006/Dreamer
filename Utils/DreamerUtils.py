import torch
import numpy as np

def gaussian_log_probability(
        x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
    ):
    """Computes the log probability of x on a given gaussian distribution.

        Args:
            x (torch.Tensor): Input to find log prob of.
            mu (torch.Tensor): Mean of gaussian distribution.
            sigma (torch.Tensor): std.dev of gaussian distribution.

        Returns:
            log_prob (torch.Tensor): Log probability of x in gaussian.
    """
    dist = torch.distributions.Normal(loc=mu, scale=sigma)
    log_prob = dist.log_prob(x)
    return log_prob

def symlog(x: torch.Tensor):
    """Symetric log function.

        Args:
            x (torch.Tensor): Tensor input to function.

        Returns:
            symlog (torch.Tensor): Symetric log of input x.
    """
    return torch.sign(x) * torch.log(1.0 + torch.abs(x))

def symlog_np(x: np.ndarray):
    """Symetric log function (numpy).

        Args:
            x (np.ndarray): Numpy array input to function.

        Returns:
            symlog (np.ndarray): Symetric log of input x.
    """
    return np.sign(x) * np.log(1.0 + np.abs(x))

def symexp(x: torch.Tensor):
    """Symetric exponential function.

        Args:
            x (torch.Tensor): Tensor input to function.

        Returns:
            symexp (torch.Tensor): Symetric exponential of input x.
    """
    # Clamp input to prevent explosion and calculate return
    x = torch.clamp(x, -20.0, 20.0)
    return torch.sign(x) * (torch.exp(torch.abs(x).float()) - 1.0)

def to_twohot(value: torch.Tensor, buckets: torch.Tensor):
    """Converts input value to a twohot encoding.

        Args:
            value (torch.Tensor): Value to convert to twohot encoding.
            buckets (torch.Tensor): Buckets to distribute value across.

        Returns:
            twohot (torch.Tensor): Twohot encoding of the input value.
    """
    # Clamp value to fit into buckets
    clipped_value = torch.clamp(max=buckets.max(), min=buckets.min(), input=value)

    # Find the first bucket below the input value
    lower_bucket_idx = torch.searchsorted(buckets, clipped_value, right=True) - 1
    lower_bucket_idx = torch.clamp(lower_bucket_idx, max=len(buckets) - 2)
    lower_bucket_val = buckets[lower_bucket_idx]

    # Get first bucket above input value
    upper_bucket_val = buckets[lower_bucket_idx + 1]

    # Calculates weight for upper and lower buckets
    weight = (clipped_value - lower_bucket_val) / (upper_bucket_val - lower_bucket_val + 1e-8)

    # Shapes tensor from (batch, seq, 1) to (batch, seq, buckets)
    twohot_shape = value.shape[:-1] + (buckets.shape[0],)
    twohot = torch.zeros(twohot_shape, dtype=torch.float32, device=value.device)

    # Create twohot vector, first call injects the lower bucket weight
    # the second call injects the upper bucker weight
    twohot = torch.scatter(twohot, dim=-1, index=lower_bucket_idx, src=(1.0 - weight))
    twohot = torch.scatter(twohot, dim=-1, index=(lower_bucket_idx + 1), src=weight)
    return twohot

def _sanitize_for_save(data_list):
    """Helper to recursively convert GPU tensors to CPU floats/arrays
        so that they can be saved.

        Args:
            data_list (torch.Tensor): Data to convert to save format. 

        Returns:
            clean_data (np.ndarray): Converted data.
    """
    clean_data = []
    for item in data_list:
        if isinstance(item, torch.Tensor):
            clean_data.append(item.detach().cpu().item())
        elif isinstance(item, list):
            # Handle list of tensors
            clean_data.append(
                [x.detach().cpu().item() if isinstance(x, torch.Tensor) else x for x in item]
            )
        else:
            clean_data.append(item)
    return np.array(clean_data)