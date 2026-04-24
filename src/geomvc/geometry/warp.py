"""Cross-view warping interfaces for GeoMVC scaffold."""

import torch
from torch import Tensor
import torch.nn.functional as F

from geomvc.models.base_predictor import MaterialPrediction


def make_normalized_grid(batch: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Create an identity sampling grid for ``grid_sample`` in [-1, 1] range."""

    ys, xs = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype),
        indexing="ij",
    )
    grid = torch.stack([xs, ys], dim=-1).unsqueeze(0).repeat(batch, 1, 1, 1)
    return grid


def compute_valid_mask(grid: Tensor) -> Tensor:
    """Compute in-bounds valid mask from normalized sampling grid."""

    x_ok = (grid[..., 0] >= -1.0) & (grid[..., 0] <= 1.0)
    y_ok = (grid[..., 1] >= -1.0) & (grid[..., 1] <= 1.0)
    return (x_ok & y_ok).float().unsqueeze(1)


def warp_material_prediction(pred: MaterialPrediction, grid: Tensor) -> tuple[MaterialPrediction, Tensor]:
    """Warp material maps using sampling grid.

    This public scaffold uses generic ``grid_sample`` with a provided grid.
    The full method should derive grids from depth + camera reprojection and
    include robust occlusion handling.
    """

    def _warp(x: Tensor) -> Tensor:
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

    warped = MaterialPrediction(
        albedo=_warp(pred.albedo),
        normal=F.normalize(_warp(pred.normal), dim=1, eps=1e-6),
        roughness=_warp(pred.roughness),
        metallic=_warp(pred.metallic),
    )
    valid_mask = compute_valid_mask(grid)
    return warped, valid_mask
