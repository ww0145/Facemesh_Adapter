"""
Abstract base class for IP adapters (TRELLIS.2 version).

Defines the shared interface: freeze base, expose trainable params,
save/load adapter weights.
"""
import torch
import torch.nn as nn
from abc import abstractmethod


class BaseAdapter(nn.Module):
    """
    Base class for all IP adapter wrappers.

    Subclasses must:
        - Set self.base (frozen base model)
        - Set self.id_proj (ImageProjModel)
        - Set self.ip_layers (ModuleList of cross-attention layers)
        - Implement forward()
    """

    def _freeze_base(self):
        """Freeze all parameters in the base model."""
        for p in self.base.parameters():
            p.requires_grad = False

    def get_trainable_params(self):
        """Return list of trainable parameters.
        If id_proj is shared across adapters, only ip_layers params are
        returned here — the caller should collect id_proj params once.
        """
        params = list(self.ip_layers.parameters())
        if not getattr(self, '_shared_proj', False):
            params = list(self.id_proj.parameters()) + params
        return params

    def get_ip_layer_params(self):
        """Return only the IP cross-attention layer parameters."""
        return list(self.ip_layers.parameters())

    @property
    def device(self):
        return self.base.device

    @property
    def dtype(self):
        return self.base.dtype

    def __getattr__(self, name):
        """Delegate attribute access to base model for pipeline compatibility."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base, name)

    @abstractmethod
    def forward(self, x, t, cond, id_cond=None, **kwargs):
        """
        Forward pass with identity conditioning.

        Args:
            x: input tensor (dense or sparse depending on stage)
            t: [B] timestep
            cond: condition tokens (Tensor or VarLenTensor)
            id_cond: [B, D] identity embedding, or None to skip adapter.
                     If None, falls back to cached id_cond from set_id_cond().
            **kwargs: additional args (e.g. concat_cond for texture stage)

        Returns:
            velocity prediction, same type as x
        """
        ...

    def set_id_cond(self, id_cond):
        """
        Cache identity embedding for use with pipeline's sampler.

        Call this before passing the adapter to pipeline.sample_shape_slat(),
        since the sampler calls forward() without id_cond argument.

        Args:
            id_cond: [B, D] identity embedding, or None to clear
        """
        self._cached_id_cond = id_cond

    def clear_id_cond(self):
        """Clear cached identity embedding."""
        self._cached_id_cond = None

    def save_adapter(self, path: str):
        """Save only adapter weights.
        If id_proj is shared, it is NOT saved here — save it separately.
        """
        state = {'ip_layers': self.ip_layers.state_dict()}
        if not getattr(self, '_shared_proj', False):
            state['id_proj'] = self.id_proj.state_dict()
        torch.save(state, path)

    def load_adapter(self, path: str):
        """Load adapter weights from checkpoint."""
        state = torch.load(path, map_location='cpu')
        self.ip_layers.load_state_dict(state['ip_layers'])
        if 'id_proj' in state and not getattr(self, '_shared_proj', False):
            self.id_proj.load_state_dict(state['id_proj'])