"""Camera utility interfaces with simplified implementations."""

from torch import Tensor
import torch


def backproject_pixels(depth: Tensor, intrinsics: Tensor) -> Tensor:
    """Backproject depth map pixels to camera-space 3D points.

    Args:
        depth: Tensor of shape [B, 1, H, W].
        intrinsics: Tensor of shape [B, 3, 3].

    Returns:
        points_cam: Tensor of shape [B, H, W, 3].

    Note:
        This public scaffold provides a simplified implementation.
        A production version may include more robust camera conventions,
        distortion handling, and numerical safeguards.
    """

    b, _, h, w = depth.shape
    ys, xs = torch.meshgrid(
        torch.arange(h, device=depth.device, dtype=depth.dtype),
        torch.arange(w, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    xs = xs.unsqueeze(0).expand(b, -1, -1)
    ys = ys.unsqueeze(0).expand(b, -1, -1)

    fx = intrinsics[:, 0, 0].view(b, 1, 1)
    fy = intrinsics[:, 1, 1].view(b, 1, 1)
    cx = intrinsics[:, 0, 2].view(b, 1, 1)
    cy = intrinsics[:, 1, 2].view(b, 1, 1)

    z = depth[:, 0]
    x = (xs - cx) * z / (fx + 1e-8)
    y = (ys - cy) * z / (fy + 1e-8)

    return torch.stack([x, y, z], dim=-1)


def project_points(points_cam: Tensor, intrinsics: Tensor) -> Tensor:
    """Project camera-space points to image pixel coordinates.

    Args:
        points_cam: Tensor [B, H, W, 3] or [B, N, 3].
        intrinsics: Tensor [B, 3, 3].

    Returns:
        pixel coordinates with last dimension 2.
    """

    is_grid = points_cam.ndim == 4
    if is_grid:
        b, h, w, _ = points_cam.shape
        pts = points_cam.view(b, h * w, 3)
    else:
        b, _, _ = points_cam.shape
        pts = points_cam

    x, y, z = pts[..., 0], pts[..., 1], pts[..., 2].clamp(min=1e-8)
    fx = intrinsics[:, 0, 0].unsqueeze(1)
    fy = intrinsics[:, 1, 1].unsqueeze(1)
    cx = intrinsics[:, 0, 2].unsqueeze(1)
    cy = intrinsics[:, 1, 2].unsqueeze(1)

    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    pixels = torch.stack([u, v], dim=-1)

    if is_grid:
        return pixels.view(b, h, w, 2)
    return pixels


def transform_points(points: Tensor, transform: Tensor) -> Tensor:
    """Apply homogeneous 4x4 transform to 3D points.

    Args:
        points: Tensor [B, H, W, 3] or [B, N, 3].
        transform: Tensor [B, 4, 4].
    """

    is_grid = points.ndim == 4
    if is_grid:
        b, h, w, _ = points.shape
        pts = points.view(b, h * w, 3)
    else:
        b, _, _ = points.shape
        pts = points

    ones = torch.ones((*pts.shape[:2], 1), device=pts.device, dtype=pts.dtype)
    pts_h = torch.cat([pts, ones], dim=-1)
    transformed = torch.bmm(pts_h, transform.transpose(1, 2))[..., :3]

    if is_grid:
        return transformed.view(b, h, w, 3)
    return transformed
