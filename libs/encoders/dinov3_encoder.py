"""
DINOv3 mean-pool identity encoder for TRELLIS.2.

Uses the pipeline's built-in DINOv3 image encoder, then mean-pools
over patch tokens to get a single identity vector.

This is the default encoder for FaceMesh-Adapter v2.
"""
import torch
from PIL import Image
from typing import Union, List
from .base import BaseIdentityEncoder


class DINOv3MeanPoolEncoder(BaseIdentityEncoder):
    """
    Identity encoder using TRELLIS.2 pipeline's DINOv3 backbone.

    pipeline.get_cond() returns [B, N_tok, 1024] DINOv3 tokens.
    We mean-pool over the token dimension to get [B, 1024].
    """

    def __init__(self, pipeline):
        """
        Args:
            pipeline: Trellis2ImageTo3DPipeline instance (provides
                      preprocess_image and get_cond).
        """
        self.pipeline = pipeline

    def encode(self, images: Union[Image.Image, List[Image.Image]],
               device: torch.device) -> torch.Tensor:
        """
        Encode image(s) → [B, 1024] identity embedding via DINOv3 mean-pool.

        Args:
            images: single PIL image or list of PIL images
            device: target device

        Returns:
            id_emb: [B, 1024]
        """
        if isinstance(images, Image.Image):
            images = [images]

        processed = [self.pipeline.preprocess_image(img) for img in images]
        with torch.no_grad():
            cond_dict = self.pipeline.get_cond(processed, resolution=512)

        cond = cond_dict['cond'].to(device)   # [B, N_tok, 1024]
        id_emb = cond.mean(dim=1)              # [B, 1024]
        return id_emb

    def encode_with_cond(self, images: Union[Image.Image, List[Image.Image]],
                         device: torch.device, resolution: int = 512):
        """
        Encode and also return the full DINOv3 condition tokens.

        Returns:
            id_emb: [B, 1024]
            cond: [B, N_tok, 1024]
            neg_cond: [B, N_tok, 1024]
        """
        if isinstance(images, Image.Image):
            images = [images]

        processed = [self.pipeline.preprocess_image(img) for img in images]
        with torch.no_grad():
            cond_dict = self.pipeline.get_cond(processed, resolution=resolution)

        cond = cond_dict['cond'].to(device)
        neg_cond = cond_dict['neg_cond'].to(device)
        id_emb = cond.mean(dim=1)
        return id_emb, cond, neg_cond

    @property
    def embed_dim(self) -> int:
        return 1024
