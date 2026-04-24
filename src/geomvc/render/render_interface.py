"""Renderer interface for public scaffold."""

from torch import Tensor

from geomvc.models.base_predictor import MaterialPrediction


class RendererInterface:
    """Placeholder differentiable renderer interface.

    The production renderer is intentionally withheld pre-publication.
    Public scaffold behavior returns albedo as a dummy rendered RGB output.
    """

    def render(
        self,
        pred: MaterialPrediction,
        depth: Tensor,
        intrinsics: Tensor,
        extrinsics: Tensor,
    ) -> Tensor:
        _ = depth, intrinsics, extrinsics
        return pred.albedo
