"""
Shape adapter trainer: trains IPAdapterSLatFlowModel on shape latents.

Adapts v1's Stage2Trainer for TRELLIS.2:
    - Uses SLatFlowModel (no U-Net, no skip connections)
    - Uses ShapeAdapterDataset
    - Shared id_proj support
    - Normalizes GT shape latent using pipeline's shape_slat_normalization
"""
import torch
import torch.nn.functional as F
from trellis2.modules import sparse as sp

from .base_trainer import BaseTrainer
from ..libs.adapters import IPAdapterSLatFlowModel
from ..libs.encoders.projections import ImageProjModel
from ..data.dataset import ShapeAdapterDataset


class ShapeTrainer(BaseTrainer):

    @property
    def run_prefix(self) -> str:
        return "shape"

    def _build_model(self, pipeline):
        cfg = self.cfg.model
        base_model = pipeline.models[cfg.flow_model_key]

        # Load normalization params from pipeline
        norm = getattr(pipeline, 'shape_slat_normalization', None)
        if norm is not None:
            self._norm_mean = torch.tensor(norm['mean'], dtype=torch.float32).to(self.device)
            self._norm_std = torch.tensor(norm['std'], dtype=torch.float32).to(self.device)
            print(f"  Shape normalization loaded: mean [{self._norm_mean.min():.2f}, {self._norm_mean.max():.2f}], "
                  f"std [{self._norm_std.min():.2f}, {self._norm_std.max():.2f}]")
        else:
            self._norm_mean = None
            self._norm_std = None
            print("  WARNING: No shape normalization found in pipeline config")

        shared_proj = ImageProjModel(
            in_dim=cfg.id_channels,
            out_dim=base_model.model_channels,
            num_tokens=cfg.num_id_tokens,
        ).to(self.device)

        model = IPAdapterSLatFlowModel(
            base_model,
            id_channels=cfg.id_channels,
            num_id_tokens=cfg.num_id_tokens,
            ip_scale=cfg.ip_scale,
            ip_interval=cfg.ip_interval,
            shared_id_proj=shared_proj,
        )
        return model, shared_proj

    def _load_dataset(self):
        dataset = ShapeAdapterDataset(
            self.cfg.data.train_dir,
            split='train',
            val_ratio=self.cfg.data.val_ratio,
        )
        sigma_min = 1e-5  # TRELLIS.2 default
        return dataset, sigma_min

    def _train_step(self, model, data, t_scalar, sigma_min):
        coords = data['coords']                                # [N, 4]
        x_0_raw = data['shape_feats']                          # [N, 32]
        id_emb = data['id_emb'].unsqueeze(0)                   # [1, D]
        cond = data['cond'].unsqueeze(0)                       # [1, N_tok, 1024]

        # Normalize GT latent (flow model trains on normalized space)
        if self._norm_mean is not None:
            x_0 = (x_0_raw - self._norm_mean) / self._norm_std
        else:
            x_0 = x_0_raw

        t = torch.tensor([t_scalar * 1000], device=self.device, dtype=torch.float32)

        # Flow matching: noisy sample + velocity target
        noise = torch.randn_like(x_0)
        x_t_feats = (1 - t_scalar) * x_0 + (sigma_min + (1 - sigma_min) * t_scalar) * noise
        v_target = (1 - sigma_min) * noise - x_0

        x_t = sp.SparseTensor(feats=x_t_feats, coords=coords)

        v_pred = model(x_t, t, cond, id_cond=id_emb)

        return F.mse_loss(v_pred.feats, v_target)