"""Dataset skeleton for GeoMVC public scaffold."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset


class MultiViewMaterialDataset(Dataset):
    """Skeleton dataset that reads a manifest or returns dummy tensors."""

    def __init__(self, root: str, manifest_path: str, image_size: tuple[int, int] = (128, 128)) -> None:
        self.root = Path(root)
        self.manifest_path = Path(manifest_path)
        self.image_size = image_size
        self.samples = self._load_manifest()

    def _load_manifest(self) -> list[dict[str, Any]]:
        if not self.manifest_path.exists():
            return [{"object_id": "dummy_object", "views": [{"view_id": "view_000"}]}]
        with self.manifest_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "objects" in data:
            return data["objects"]
        if isinstance(data, list):
            return data
        return [data]

    def __len__(self) -> int:
        return len(self.samples)

    def _load_rgb(self, path: Path) -> Tensor:
        if not path.exists():
            return torch.rand(3, *self.image_size)
        img = Image.open(path).convert("RGB").resize(self.image_size[::-1])
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def _load_mask(self, path: Path) -> Tensor:
        if not path.exists():
            return torch.ones(1, *self.image_size)
        img = Image.open(path).convert("L").resize(self.image_size[::-1])
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr)[None, ...]

    def _load_depth(self, path: Path) -> Tensor:
        if not path.exists():
            return torch.ones(1, *self.image_size)
        arr = np.load(path).astype(np.float32)
        if arr.shape != self.image_size:
            arr = np.resize(arr, self.image_size)
        return torch.from_numpy(arr)[None, ...]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        view = sample.get("views", [{}])[0]

        rgb = self._load_rgb(self.root / view.get("image", "missing_rgb.png"))
        depth = self._load_depth(self.root / view.get("depth", "missing_depth.npy"))
        mask = self._load_mask(self.root / view.get("mask", "missing_mask.png"))

        intrinsics = torch.tensor(
            view.get("camera", {}).get(
                "intrinsics",
                [[100.0, 0.0, self.image_size[1] / 2], [0.0, 100.0, self.image_size[0] / 2], [0.0, 0.0, 1.0]],
            ),
            dtype=torch.float32,
        )
        extrinsics = torch.tensor(
            view.get("camera", {}).get(
                "extrinsics",
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            ),
            dtype=torch.float32,
        )

        targets = {
            "albedo": torch.zeros_like(rgb),
            "normal": torch.cat([torch.zeros(2, *self.image_size), torch.ones(1, *self.image_size)], dim=0),
            "roughness": torch.full((1, *self.image_size), 0.5),
            "metallic": torch.zeros(1, *self.image_size),
        }

        return {
            "rgb": rgb,
            "depth": depth,
            "mask": mask,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "targets": targets,
        }
