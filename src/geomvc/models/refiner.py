"""One-step refiner stub for pre-publication GeoMVC scaffold."""

from torch import Tensor, nn
import torch
import torch.nn.functional as F

from .base_predictor import MaterialPrediction


class OneStepRefinerStub(nn.Module):
    """Placeholder residual refiner.

    This is not the real production/diffusion refiner from the paper.
    It only demonstrates input-output interfaces for public release.
    """

    def __init__(self, in_channels: int = 11, hidden_channels: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 8, kernel_size=1),
        )

    def forward(self, rgb: Tensor, base_pred: MaterialPrediction) -> MaterialPrediction:
        x = torch.cat(
            [
                rgb,
                base_pred.albedo,
                base_pred.normal,
                base_pred.roughness,
                base_pred.metallic,
            ],
            dim=1,
        )
        residual = self.net(x)

        albedo = torch.clamp(base_pred.albedo + residual[:, 0:3], min=0.0, max=1.0)
        normal = F.normalize(base_pred.normal + residual[:, 3:6], dim=1, eps=1e-6)
        roughness = torch.clamp(base_pred.roughness + residual[:, 6:7], min=0.0, max=1.0)
        metallic = torch.clamp(base_pred.metallic + residual[:, 7:8], min=0.0, max=1.0)

        return MaterialPrediction(albedo=albedo, normal=normal, roughness=roughness, metallic=metallic)
