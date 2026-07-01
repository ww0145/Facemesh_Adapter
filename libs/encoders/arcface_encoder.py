"""
ArcFace identity encoder (placeholder).

TODO: Implement with insightface or onnx runtime.
"""
import torch
from PIL import Image
from typing import Union, List
from .base import BaseIdentityEncoder


class ArcFaceEncoder(BaseIdentityEncoder):
    """
    Identity encoder using ArcFace face recognition model.

    Outputs a 512-d identity embedding optimized for face identity
    discrimination. Unlike DINOv3 (general visual features),
    ArcFace is specifically trained for face recognition.

    TODO:
        - Load pretrained ArcFace model (e.g. insightface buffalo_l)
        - Implement face detection + alignment preprocessing
        - Handle cases where no face is detected
    """

    def __init__(self, model_name: str = "buffalo_l"):
        self.model_name = model_name
        # TODO: load model
        raise NotImplementedError("ArcFace encoder not yet implemented")

    def encode(self, images: Union[Image.Image, List[Image.Image]],
               device: torch.device) -> torch.Tensor:
        raise NotImplementedError

    @property
    def embed_dim(self) -> int:
        return 512
