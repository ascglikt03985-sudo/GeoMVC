# GeoMVC (Pre-Publication Public Scaffold)

GeoMVC is a **Geometry-Aligned Multi-View Consistency** framework for PBR material decomposition.

> **Important notice**
>
> This repository is a **pre-publication public scaffold**.
> It provides interfaces, configuration files, data schema, and a runnable dummy pipeline only.
> Full training code, exact model architecture, private data processing scripts, checkpoints, and the production differentiable renderer will be released **after publication**.

## Method Overview

GeoMVC takes multi-view inputs:
- RGB images
- Depth maps
- Foreground masks
- Camera intrinsics and extrinsics

And predicts per-view material components:
- Albedo
- Normal
- Roughness
- Metallic

## Three-Stage Pipeline

1. **Stage I: Base view-space PBR prediction**  
   A base predictor produces initial material maps from RGB/depth/mask.

2. **Stage II: Physics-guided refinement with render-back supervision**  
   A refinement module improves predictions with a render-back consistency placeholder.

3. **Stage III: Geometry-aligned multi-view consistency**  
   Multi-view warping, occlusion-aware reliability weighting, SigmaNet, and component-aware consistency losses are exposed through public interfaces/stubs.

## Repository Layout

```text
GeoMVC/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── configs/
│   ├── base.yaml
│   ├── stage1_base.yaml
│   ├── stage2_refine.yaml
│   ├── stage3_mvc.yaml
│   └── data_schema.yaml
├── src/geomvc/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_predictor.py
│   │   ├── refiner.py
│   │   └── sigmanet.py
│   ├── geometry/
│   │   ├── __init__.py
│   │   ├── camera.py
│   │   └── warp.py
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── material_losses.py
│   │   └── mvc_losses.py
│   ├── render/
│   │   ├── __init__.py
│   │   └── render_interface.py
│   └── data/
│       ├── __init__.py
│       └── dataset.py
├── scripts/
│   ├── prepare_dummy_data.py
│   ├── train.py
│   ├── infer.py
│   └── evaluate.py
└── data_schema/
    └── sample_manifest.json
```

## Quick Start

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Prepare demo data

```bash
python scripts/prepare_dummy_data.py
```

### 3) Run dummy training entry

```bash
python scripts/train.py --config configs/stage1_base.yaml
```

### 4) Run dummy inference

```bash
python scripts/infer.py --config configs/stage3_mvc.yaml
```

### 5) Show evaluation hooks

```bash
python scripts/evaluate.py
```

## Data Format

Use `data_schema/sample_manifest.json` as a reference manifest for one object with multiple views.
Expected per-view fields include image/depth/mask paths, camera intrinsics/extrinsics, and optional targets.

You can also see schema notes in `configs/data_schema.yaml`.

## Release Policy

This public scaffold intentionally withholds publication-sensitive assets before paper release, including:
- Full training code and full training loop details
- Exact architecture design and final paper hyperparameters
- Private/production data preprocessing pipeline
- Model checkpoints
- Production differentiable renderer implementation

After publication, we plan to release additional components in stages, subject to data/license constraints.

## Citation

If you find this scaffold useful, please cite:

```bibtex
@article{liu_geomvc_tba,
  title   = {GeoMVC: Geometry-Aligned Multi-View Consistency for PBR Material Decomposition},
  author  = {Sanyuan Liu and collaborators},
  journal = {TBD},
  year    = {TBD}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
