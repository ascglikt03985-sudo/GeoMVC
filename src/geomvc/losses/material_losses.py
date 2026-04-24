"""Material prediction losses (public scaffold version)."""

from torch import Tensor
import torch

from geomvc.models.base_predictor import MaterialPrediction


def masked_l1(pred: Tensor, target: Tensor, mask: Tensor, eps: float = 1e-6) -> Tensor:
    """Masked L1 loss with safe normalization."""

    diff = (pred - target).abs() * mask
    denom = mask.sum().clamp(min=eps)
    return diff.sum() / denom


def normal_cosine_loss(pred_normal: Tensor, target_normal: Tensor, mask: Tensor, eps: float = 1e-6) -> Tensor:
    """Cosine loss for normal vectors."""

    cos = (pred_normal * target_normal).sum(dim=1, keepdim=True).clamp(min=-1.0, max=1.0)
    loss = (1.0 - cos) * mask
    denom = mask.sum().clamp(min=eps)
    return loss.sum() / denom


def base_supervision_loss(
    pred: MaterialPrediction,
    targets: dict[str, Tensor],
    mask: Tensor,
    weights: dict[str, float],
) -> Tensor:
    """Aggregate placeholder supervision losses for material channels."""

    total = torch.tensor(0.0, device=pred.albedo.device)
    if "albedo" in targets:
        total = total + weights.get("albedo_l1", 1.0) * masked_l1(pred.albedo, targets["albedo"], mask)
    if "normal" in targets:
        total = total + weights.get("normal_cosine", 1.0) * normal_cosine_loss(pred.normal, targets["normal"], mask)
    if "roughness" in targets:
        total = total + weights.get("roughness_l1", 1.0) * masked_l1(pred.roughness, targets["roughness"], mask)
    if "metallic" in targets:
        total = total + weights.get("metallic_l1", 1.0) * masked_l1(pred.metallic, targets["metallic"], mask)
    return total


def render_back_loss(rendered_rgb: Tensor, observed_rgb: Tensor, mask: Tensor) -> Tensor:
    """Placeholder render-back supervision loss."""

    return masked_l1(rendered_rgb, observed_rgb, mask)
