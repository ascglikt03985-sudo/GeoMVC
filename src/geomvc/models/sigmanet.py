"""SigmaNet reliability predictor stub for GeoMVC scaffold."""

import torch
from torch import Tensor, nn


class SigmaNetStub(nn.Module):
    """Predicts reliability from depth discrepancy and validity mask."""

    def __init__(self, in_channels: int = 2, hidden_channels: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, depth_discrepancy: Tensor, valid_mask: Tensor) -> Tensor:
        x = torch.cat([depth_discrepancy, valid_mask], dim=1)
        sigma = self.net(x)
        reliability = torch.exp(-torch.relu(sigma)) * valid_mask
        return reliability
