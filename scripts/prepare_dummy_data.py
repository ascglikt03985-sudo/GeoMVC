"""Prepare minimal dummy data for GeoMVC scaffold quick start."""

from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    base = Path("data/demo_multiview/demo_object_001")
    base.mkdir(parents=True, exist_ok=True)

    h, w = 128, 128
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 0] = np.linspace(0, 255, w, dtype=np.uint8)
    rgb[..., 1] = np.linspace(255, 0, h, dtype=np.uint8)[:, None]
    rgb[..., 2] = 128

    mask = np.ones((h, w), dtype=np.uint8) * 255
    depth = np.full((h, w), 1.0, dtype=np.float32)

    Image.fromarray(rgb).save(base / "view_000_rgb.png")
    Image.fromarray(mask).save(base / "view_000_mask.png")
    np.save(base / "view_000_depth.npy", depth)

    print(f"Dummy data written to: {base}")


if __name__ == "__main__":
    main()
