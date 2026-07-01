"""
IP cross-attention layers for identity token injection (TRELLIS.2 version).

Design follows original 2D IP-Adapter:
    - Q is borrowed from frozen cross-attention (via hook, stored in _cached_q)
    - Only K and V projections are trainable
    - Optional out_proj for ablation (use_out_proj flag)

Two variants:
    - DenseIPCrossAttnLayer:  for Stage 1 (SS) dense [B, N, dim] tensors
    - SparseIPCrossAttnLayer: for Stage 2/3 (Shape/Texture) SparseTensor inputs
"""
import torch
import torch.nn as nn


class DenseIPCrossAttnLayer(nn.Module):
    """
    IP-Adapter cross-attention for dense 3D volume features (SS stage).

    Q is borrowed from frozen cross-attention (set via _cached_q by hook).
    Only K and V are trainable. Optional out_proj for ablation.

    Input:  x [B, N, dim], id_tokens [B, T, dim]
    Output: x + scale * IP_Attn(frozen_Q, K_ip, V_ip)
    """

    def __init__(self, dim: int, num_heads: int, scale: float = 1.0,
                 use_out_proj: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale_factor = scale
        self.attn_scale = self.head_dim ** -0.5
        self.use_out_proj = use_out_proj

        # Trainable: only K, V (and optionally out_proj)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        if use_out_proj:
            self.to_out = nn.Linear(dim, dim, bias=False)
            nn.init.zeros_(self.to_out.weight)

        # Populated by hook before forward() is called
        self._cached_q = None

    def forward(self, x: torch.Tensor, id_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, dim] dense features (for residual connection)
            id_tokens: [B, T, dim] identity tokens from ImageProjModel

        Returns:
            [B, N, dim] features with identity information injected
        """
        q = self._cached_q
        assert q is not None, (
            "Q not captured — ensure hook is registered on frozen cross_attn"
        )
        self._cached_q = None  # clear after use

        B, N, _ = x.shape
        T = id_tokens.shape[1]
        H, D = self.num_heads, self.head_dim

        # q: [B, N, dim] from frozen cross_attn.to_q
        q = q.float().view(B, N, H, D).permute(0, 2, 1, 3)
        k = self.to_k(id_tokens.float()).float().view(B, T, H, D).permute(0, 2, 1, 3)
        v = self.to_v(id_tokens.float()).float().view(B, T, H, D).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.attn_scale    # [B, H, N, T]
        attn = attn.softmax(dim=-1)
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, self.dim)

        if self.use_out_proj:
            out = self.to_out(out)

        return (x.float() + self.scale_factor * out).to(x.dtype)


class SparseIPCrossAttnLayer(nn.Module):
    """
    IP-Adapter cross-attention for sparse voxel features (Shape/Texture stages).

    Q is borrowed from frozen cross-attention (set via _cached_q by hook).
    Only K and V are trainable. Optional out_proj for ablation.

    Input:  x (SparseTensor with .feats [N_total, dim]), id_tokens [B, T, dim]
    Output: SparseTensor with updated features
    """

    def __init__(self, dim: int, num_heads: int, scale: float = 1.0,
                 use_out_proj: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale_factor = scale
        self.attn_scale = self.head_dim ** -0.5
        self.use_out_proj = use_out_proj

        # Trainable: only K, V (and optionally out_proj)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        if use_out_proj:
            self.to_out = nn.Linear(dim, dim, bias=False)
            nn.init.zeros_(self.to_out.weight)

        # Populated by hook before forward() is called
        self._cached_q = None

    def forward(self, x, id_tokens: torch.Tensor):
        """
        Args:
            x: SparseTensor with .feats [N_total, dim]
            id_tokens: [B, T, dim] identity tokens

        Returns:
            SparseTensor with updated features
        """
        q = self._cached_q
        assert q is not None, (
            "Q not captured — ensure hook is registered on frozen cross_attn"
        )
        self._cached_q = None  # clear after use

        batch_ids = x.coords[:, 0].long()
        id_expanded = id_tokens[batch_ids]  # [N_total, T, dim]

        N, T = q.shape[0], id_expanded.shape[1]
        H, D = self.num_heads, self.head_dim

        # q: [N_total, dim] from frozen cross_attn.to_q
        q = q.float().view(N, H, D)
        k = self.to_k(id_expanded.float()).float().view(N, T, H, D).permute(0, 2, 1, 3)
        v = self.to_v(id_expanded.float()).float().view(N, T, H, D).permute(0, 2, 1, 3)

        attn = torch.einsum('nhd,nhtd->nht', q, k) * self.attn_scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum('nht,nhtd->nhd', attn, v).reshape(N, self.dim)

        if self.use_out_proj:
            out = self.to_out(out)

        new_feats = x.feats.float() + self.scale_factor * out
        return x.replace(new_feats.to(x.feats.dtype))