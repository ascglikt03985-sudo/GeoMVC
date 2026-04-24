"""Base predictor stub for pre-publication GeoMVC scaffold."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class MaterialPrediction:
    """Container for per-view PBR predictions."""

    albedo: Tensor
    normal: Tensor
    roughness: Tensor
    metallic: Tensor


class BasePredictorStub(nn.Module):
    """A simplified ConvNet placeholder for Stage-I base prediction.

    This is intentionally lightweight and does not reflect the full paper model.
    """

    def __init__(self, in_channels: int = 5, hidden_channels: int = 32) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(hidden_channels, 8, kernel_size=1)

    def forward(self, rgb: Tensor, depth: Tensor, mask: Tensor) -> MaterialPrediction:
        x = torch.cat([rgb, depth, mask], dim=1)
        feat = self.backbone(x)
        out = self.head(feat)

        albedo = torch.sigmoid(out[:, 0:3])
        normal = F.normalize(out[:, 3:6], dim=1, eps=1e-6)
        roughness = torch.sigmoid(out[:, 6:7])
        metallic = torch.sigmoid(out[:, 7:8])

        return MaterialPrediction(
            albedo=albedo,
            normal=normal,
            roughness=roughness,
            metallic=metallic,
        )
