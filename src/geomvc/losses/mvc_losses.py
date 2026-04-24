"""Multi-view consistency losses (public scaffold placeholders)."""

from torch import Tensor
import torch


def chroma_consistency_loss(src_albedo: Tensor, warped_albedo: Tensor, reliability: Tensor, eps: float = 1e-6) -> Tensor:
    """Reliability-weighted chroma consistency L1 loss."""

    diff = (src_albedo - warped_albedo).abs() * reliability
    denom = reliability.sum().clamp(min=eps)
    return diff.sum() / denom


def high_frequency_loss(src: Tensor, ref: Tensor, reliability: Tensor, eps: float = 1e-6) -> Tensor:
    """High-frequency placeholder loss.

    This is a lightweight approximation and intentionally not the full paper loss.
    """

    src_dx = src[..., :, 1:] - src[..., :, :-1]
    ref_dx = ref[..., :, 1:] - ref[..., :, :-1]
    rel = reliability[..., :, 1:]
    diff = (src_dx - ref_dx).abs() * rel
    denom = rel.sum().clamp(min=eps)
    return diff.sum() / denom


def mvc_loss(
    src_albedo: Tensor,
    warped_albedo: Tensor,
    src_roughness: Tensor,
    warped_roughness: Tensor,
    reliability: Tensor,
    weights: dict[str, float],
) -> Tensor:
    """Aggregate MVC loss with reliability weighting."""

    chroma = chroma_consistency_loss(src_albedo, warped_albedo, reliability)
    hf = high_frequency_loss(src_roughness, warped_roughness, reliability)
    return weights.get("chroma", 1.0) * chroma + weights.get("high_frequency", 0.1) * hf
