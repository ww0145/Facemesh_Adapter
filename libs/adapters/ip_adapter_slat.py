"""
Stage 2/3 IP-Adapter: wraps TRELLIS.2 SLatFlowModel (sparse voxels).

Injects identity tokens via SparseIPCrossAttnLayer every `ip_interval`
transformer blocks. Used for both Shape (stage 2) and Texture (stage 3).

Design follows original 2D IP-Adapter:
    - Q is borrowed from frozen cross-attention via forward hook
    - IP layers only have trainable K, V (+ optional out_proj)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from trellis2.modules.utils import manual_cast
from trellis2.modules import sparse as sp

from .base import BaseAdapter
from .cross_attention import SparseIPCrossAttnLayer
from ..encoders.projections import ImageProjModel


class IPAdapterSLatFlowModel(BaseAdapter):
    """
    Stage 2/3 adapter for TRELLIS.2 SLatFlowModel.

    Works for both Shape and Texture flow models:
        - Shape: forward(x, t, cond, id_cond=...)
        - Texture: forward(x, t, cond, id_cond=..., concat_cond=shape_latent)

    Args:
        base_model: frozen SLatFlowModel instance
        id_channels: dimension of input identity embedding
        num_id_tokens: number of identity tokens produced by ImageProjModel
        ip_scale: scaling factor for cross-attention residual
        ip_interval: insert IP layer every N blocks (default 2)
        use_out_proj: whether IP layers include out_proj (for ablation)
        shared_id_proj: shared ImageProjModel instance, or None to create one
    """

    def __init__(self, base_model, id_channels: int = 1024,
                 num_id_tokens: int = 4, ip_scale: float = 1.0,
                 ip_interval: int = 1, use_out_proj: bool = False,
                 shared_id_proj: ImageProjModel = None):
        super().__init__()
        self.base = base_model
        dim = base_model.model_channels       # 1536
        num_heads = base_model.num_heads       # 12

        self._freeze_base()

        if shared_id_proj is not None:
            self.id_proj = shared_id_proj
            self._shared_proj = True
        else:
            self.id_proj = ImageProjModel(
                in_dim=id_channels, out_dim=dim, num_tokens=num_id_tokens
            )
            self._shared_proj = False

        # IP layer indices: every ip_interval blocks
        self.ip_block_indices = set(
            range(ip_interval - 1, base_model.num_blocks, ip_interval)
        )
        num_ip_layers = len(self.ip_block_indices)

        self.ip_layers = nn.ModuleList([
            SparseIPCrossAttnLayer(
                dim, num_heads, scale=ip_scale, use_out_proj=use_out_proj
            )
            for _ in range(num_ip_layers)
        ])

        # Register hooks to capture Q from frozen cross-attention
        self._hooks = []
        self._register_q_hooks()

    def _register_q_hooks(self):
        """
        For each IP layer, register a forward hook on the corresponding
        frozen block's cross_attn to capture Q = cross_attn.to_q(normed_input).
        """
        ip_idx = 0
        for i, block in enumerate(self.base.blocks):
            if i in self.ip_block_indices:
                cross_attn = block.cross_attn  # MultiHeadAttention (type="cross")
                ip_layer = self.ip_layers[ip_idx]
                hook_fn = self._make_q_hook(cross_attn, ip_layer)
                handle = cross_attn.register_forward_hook(hook_fn)
                self._hooks.append(handle)
                ip_idx += 1

    @staticmethod
    def _make_q_hook(cross_attn_module, ip_layer):
        """
        Create a forward hook that captures Q from frozen cross-attention.

        In ModulatedTransformerCrossBlock._forward:
            h = self.norm2(x)                    # norm
            h = self.cross_attn(h, context)      # ← hook fires here

        So input[0] to cross_attn is already norm2(x).
        We compute Q = cross_attn.to_q(input[0]) and store it for the IP layer.

        Note: for sparse blocks, input[0].feats gives [N_total, dim].
              for dense blocks, input[0] is [B, L, dim].
        """
        def hook_fn(module, input, output):
            h_normed = input[0]
            # Handle both SparseTensor and dense Tensor
            feats = h_normed.feats if hasattr(h_normed, 'feats') else h_normed
            q = module.to_q(feats)
            ip_layer._cached_q = q.detach()
        return hook_fn

    def remove_hooks(self):
        """Remove all registered hooks (for clean teardown)."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def forward(self, x, t, cond, id_cond=None, concat_cond=None, **kwargs):
        """
        Forward pass replicating SLatFlowModel.forward with identity
        cross-attention inserted every ip_interval blocks.

        Args:
            x: SparseTensor input
            t: [B] timestep
            cond: [B, N_tok, C_cond] condition tokens, or list of tensors
            id_cond: [B, D] identity embedding, or None
            concat_cond: SparseTensor to concatenate (texture stage uses
                         shape latent here)

        Returns:
            SparseTensor velocity prediction
        """
        if id_cond is None:
            id_cond = getattr(self, '_cached_id_cond', None)
        if id_cond is None:
            return self.base(x, t, cond, concat_cond=concat_cond, **kwargs)

        id_tokens = self.id_proj(id_cond.float())  # [B, T, 1536]

        # --- Replicate SLatFlowModel.forward ---
        if concat_cond is not None:
            x = sp.sparse_cat([x, concat_cond], dim=-1)
        if isinstance(cond, list):
            cond = sp.VarLenTensor.from_tensor_list(cond)

        h = self.base.input_layer(x)
        h = manual_cast(h, self.base.dtype)
        t_emb = self.base.t_embedder(t)
        if self.base.share_mod:
            t_emb = self.base.adaLN_modulation(t_emb)
        t_emb = manual_cast(t_emb, self.base.dtype)
        cond = manual_cast(cond, self.base.dtype)

        if self.base.pe_mode == "ape":
            pe = self.base.pos_embedder(h.coords[:, 1:])
            h = h + manual_cast(pe, self.base.dtype)

        ip_idx = 0
        for i, block in enumerate(self.base.blocks):
            # Hook fires inside block() → captures Q into ip_layer._cached_q
            h = block(h, t_emb, cond)
            if i in self.ip_block_indices:
                h = self.ip_layers[ip_idx](h, id_tokens)
                ip_idx += 1

        h = manual_cast(h, x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.base.out_layer(h)
        return h