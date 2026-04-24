"""Loss utilities for GeoMVC scaffold."""

from .material_losses import base_supervision_loss, masked_l1, normal_cosine_loss, render_back_loss
from .mvc_losses import chroma_consistency_loss, high_frequency_loss, mvc_loss

__all__ = [
    "masked_l1",
    "normal_cosine_loss",
    "base_supervision_loss",
    "render_back_loss",
    "chroma_consistency_loss",
    "high_frequency_loss",
    "mvc_loss",
]
