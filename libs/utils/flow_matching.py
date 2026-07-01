"""
Flow matching utilities for adapter training.

Provides time sampling, noise interpolation, and velocity target
computation following the conditional flow matching formulation
used by TRELLIS.2.
"""
import torch


def sample_t_logit_normal(mean: float = 0.0, std: float = 1.0) -> float:
    """Sample timestep from LogitNormal distribution (consistent with TRELLIS training)."""
    return torch.sigmoid(torch.randn(1) * std + mean).item()


def build_noisy_sample(x_0: torch.Tensor, t: float, sigma_min: float = 1e-5):
    """
    Construct noisy sample x_t and velocity target v for flow matching.

    Args:
        x_0: clean sample, any shape [...]
        t: scalar timestep in [0, 1]
        sigma_min: minimum noise scale

    Returns:
        x_t: noisy sample, same shape as x_0
        noise: sampled noise
        v_target: velocity target = (1 - sigma_min) * noise - x_0
    """
    noise = torch.randn_like(x_0)
    x_t = (1 - t) * x_0 + (sigma_min + (1 - sigma_min) * t) * noise
    v_target = (1 - sigma_min) * noise - x_0
    return x_t, noise, v_target
