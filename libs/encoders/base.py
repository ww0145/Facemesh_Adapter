"""
Abstract base class for identity encoders.

All identity encoders must implement:
    encode(image) → [B, D] identity embedding tensor
"""
from abc import ABC, abstractmethod
import torch
from PIL import Image
from typing import Union, List


class BaseIdentityEncoder(ABC):
    """
    Interface for identity feature extraction.

    Subclasses implement `encode()` which takes one or more PIL images
    and returns a fixed-size identity embedding.
    """

    @abstractmethod
    def encode(self, images: Union[Image.Image, List[Image.Image]],
               device: torch.device) -> torch.Tensor:
        """
        Extract identity embedding from image(s).

        Args:
            images: single PIL image or list of PIL images
            device: target device for output tensor

        Returns:
            id_emb: [B, D] identity embedding
        """
        ...

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """Dimensionality of the output identity embedding."""
        ...
