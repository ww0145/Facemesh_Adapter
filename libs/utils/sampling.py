"""
Euler ODE sampler for Shape and Texture stages.

Adapted from v1 to work with TRELLIS.2's SLatFlowModel.
SS stage uses the pipeline's built-in sampler (no adapter).
"""
import torch
import numpy as np
from tqdm import tqdm

from trellis2.modules import sparse as sp


def build_time_schedule(steps: int, rescale_t: float = 3.0):
    """
    Build time sequence from t=1 to t=0 with optional rescaling.

    Args:
        steps: number of integration steps
        rescale_t: rescaling factor (TRELLIS default = 3.0)

    Returns:
        t_seq: np.ndarray of shape [steps + 1]
    """
    t_seq = np.linspace(1, 0, steps + 1)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
    return t_seq


def euler_sample_sparse(model, coords, cond, id_emb, device,
                        steps=12, rescale_t=3.0,
                        concat_cond=None, verbose=True):
    """
    Euler ODE sampling for Shape/Texture stage (sparse voxels).

    Args:
        model: IPAdapterSLatFlowModel instance
        coords: [N, 4] int sparse coordinates (with batch index)
        cond: [B, N_tok, 1024] condition tokens
        id_emb: [B, D] identity embedding, or None
        device: torch device
        steps: number of Euler steps
        rescale_t: time rescaling factor
        concat_cond: SparseTensor for texture stage (shape latent)
        verbose: show progress bar

    Returns:
        SparseTensor with denoised features
    """
    base = model.base if hasattr(model, 'base') else model
    t_seq = build_time_schedule(steps, rescale_t)

    x = torch.randn(coords.shape[0], base.out_channels, device=device)

    iterator = tqdm(range(steps), desc="Sampling") if verbose else range(steps)
    for i in iterator:
        t_val = float(t_seq[i])
        t_prev = float(t_seq[i + 1])
        t_tensor = torch.tensor([t_val * 1000], device=device, dtype=torch.float32)

        x_sparse = sp.SparseTensor(feats=x, coords=coords.to(device))
        v = model(
            x_sparse, t_tensor, cond,
            id_cond=id_emb,
            concat_cond=concat_cond,
        )
        x = x - (t_val - t_prev) * v.feats

        if torch.isnan(x).any():
            print(f"NaN at step {i}!")
            return None

    return sp.SparseTensor(feats=x, coords=coords.to(device))
