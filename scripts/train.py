"""Dummy training entry for GeoMVC public scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import yaml
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geomvc.models import BasePredictorStub, OneStepRefinerStub, SigmaNetStub


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


def set_trainable(module: torch.nn.Module, trainable: bool) -> None:
    for p in module.parameters():
        p.requires_grad = trainable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/stage1_base.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Loaded stage: {cfg['project']['stage']}")

    base = BasePredictorStub(**cfg["model"]["base_predictor"])
    refiner = OneStepRefinerStub(**cfg["model"]["refiner"])
    sigmanet = SigmaNetStub(**cfg["model"]["sigmanet"])

    freeze = cfg.get("freeze", {})
    set_trainable(base, not freeze.get("base_predictor", False))
    set_trainable(refiner, not freeze.get("refiner", False))
    set_trainable(sigmanet, not freeze.get("sigmanet", False))

    print("Freeze policy:")
    print(f"  base_predictor frozen: {freeze.get('base_predictor', False)}")
    print(f"  refiner frozen: {freeze.get('refiner', False)}")
    print(f"  sigmanet frozen: {freeze.get('sigmanet', False)}")

    h, w = cfg["data"]["image_size"]
    rgb = torch.rand(1, 3, h, w)
    depth = torch.rand(1, 1, h, w)
    mask = torch.ones(1, 1, h, w)

    base_pred = base(rgb, depth, mask)
    refined = refiner(rgb, base_pred)

    depth_discrepancy = torch.rand(1, 1, h, w)
    valid_mask = mask
    reliability = sigmanet(depth_discrepancy, valid_mask)

    print(f"albedo shape: {tuple(refined.albedo.shape)}")
    print(f"normal shape: {tuple(refined.normal.shape)}")
    print(f"roughness shape: {tuple(refined.roughness.shape)}")
    print(f"metallic shape: {tuple(refined.metallic.shape)}")
    print(f"reliability shape: {tuple(reliability.shape)}")


if __name__ == "__main__":
    main()
