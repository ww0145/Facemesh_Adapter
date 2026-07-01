"""
Texture adapter trainer (placeholder).

Implement after shape adapter training works.
Will be similar to ShapeTrainer but with:
    - tex_slat_flow_model_512 as base
    - TextureAdapterDataset
    - concat_cond (shape latent conditions texture)
"""
from .base_trainer import BaseTrainer


class TextureTrainer(BaseTrainer):

    @property
    def run_prefix(self) -> str:
        return "texture"

    def _build_model(self, pipeline):
        raise NotImplementedError("TextureTrainer: implement after shape adapter works")

    def _load_dataset(self):
        raise NotImplementedError("TextureTrainer: implement after shape adapter works")

    def _train_step(self, model, data, t_scalar, sigma_min):
        raise NotImplementedError("TextureTrainer: implement after shape adapter works")
