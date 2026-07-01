"""
Projection module: identity embedding → cross-attention tokens.

Maps a single identity vector [B, in_dim] to a sequence of tokens
[B, num_tokens, out_dim] for injection via cross-attention.
"""
import torch
import torch.nn as nn


class ImageProjModel(nn.Module):
    """
    MLP projection: [B, in_dim] → [B, num_tokens, out_dim].

    Architecture: Linear → GELU → Linear → Reshape → LayerNorm

    Args:
        in_dim:     input identity embedding dimension
                    (1024 for DINOv3, 512 for ArcFace)
        out_dim:    output token dimension (must match base model's
                    model_channels, i.e. 1536 for TRELLIS.2-4B)
        num_tokens: number of identity tokens to produce
    """

    def __init__(self, in_dim: int = 1024, out_dim: int = 1536, num_tokens: int = 4):
        super().__init__()
        self.num_tokens = num_tokens
        self.out_dim = out_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim * num_tokens),
            nn.GELU(),
            nn.Linear(out_dim * num_tokens, out_dim * num_tokens),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_dim] identity embedding

        Returns:
            tokens: [B, num_tokens, out_dim]
        """
        x = self.proj(x)
        x = x.view(-1, self.num_tokens, self.out_dim)
        return self.norm(x)
