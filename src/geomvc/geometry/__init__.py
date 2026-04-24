"""Geometry utilities for GeoMVC scaffold."""

from .camera import backproject_pixels, project_points, transform_points
from .warp import compute_valid_mask, make_normalized_grid, warp_material_prediction

__all__ = [
    "backproject_pixels",
    "project_points",
    "transform_points",
    "make_normalized_grid",
    "compute_valid_mask",
    "warp_material_prediction",
]
