"""Dummy inference script for GeoMVC public scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geomvc.models import BasePredictorStub, OneStepRefinerStub


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base_path = cfg.get("base_config")
    if base_path:
        with open(base_path, "r", encoding="utf-8") as f:
            base = yaml.safe_load(f)
        cfg = deep_merge(base, cfg)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/stage3_mvc.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    h, w = cfg["data"]["image_size"]

    base = BasePredictorStub(**cfg["model"]["base_predictor"])
    refiner = OneStepRefinerStub(**cfg["model"]["refiner"])

    rgb = torch.rand(1, 3, h, w)
    depth = torch.rand(1, 1, h, w)
    mask = torch.ones(1, 1, h, w)

    pred = base(rgb, depth, mask)
    pred = refiner(rgb, pred)

    print("Dummy inference output shapes:")
    print(f"  albedo: {tuple(pred.albedo.shape)}")
    print(f"  normal: {tuple(pred.normal.shape)}")
    print(f"  roughness: {tuple(pred.roughness.shape)}")
    print(f"  metallic: {tuple(pred.metallic.shape)}")


if __name__ == "__main__":
    main()
