# Image-to-Point-Cloud 2D Super-Resolution Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a diffusion-based pipeline that converts blurred microscopy images (WF/SIM) into high-resolution probability density maps, from which precise point clouds can be sampled.

**Architecture:** Use the existing EMDiffuse real-space diffusion model (DDPM with UNet backbone) to learn the mapping from low-resolution microscopy images to probability density maps. The pipeline: Point Cloud CSV → Simulated Images (WF/SIM/STED/Density) → Train Diffusion → Inference → Sample Points from Density → Evaluate.

**Tech Stack:** PyTorch 2.9.1, CUDA 13.0, Python 3.11, tifffile, scipy, numpy, pandas

---

## Overview

### Physical Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Image size | 1024×1024 pixels | High-res density maps |
| Pixel size | 25 nm/pixel | FOV = 25.6 µm |
| WF PSF FWHM | 300 nm (σ ≈ 5.1 px) | Widefield microscopy |
| SIM PSF FWHM | 120 nm (σ ≈ 2.0 px) | Structured illumination |
| STED PSF FWHM | 50 nm (σ ≈ 0.85 px) | Stimulated emission depletion |
| Density kernel σ | 1 px (25 nm) | Ground truth localization |

### Data Flow

```
Point Cloud (CSV, nm) → 2D Projection (ignore z)
    ↓
Probability Density Map (1024×1024, normalized)
    ↓
Simulated WF/SIM/STED (convolve with PSF + noise)
    ↓
Training Pairs: (WF, Density) or (SIM, Density)
    ↓
Diffusion Model Training (real-space DDPM)
    ↓
Inference: WF/SIM → Predicted Density Map
    ↓
Sampling: Density → Point Cloud (multinomial sampling)
    ↓
Evaluation: Compare with ground truth
```

### Directory Structure

```
/data0/djx/EMDiffuse/
├── images/
│   └── microtubules/
│       ├── density/     # Ground truth density maps (1024×1024 TIFF)
│       ├── wf/          # Simulated widefield images
│       ├── sim/         # Simulated SIM images
│       ├── sted/        # Simulated STED images
│       └── metadata.json
├── training/
│   ├── wf2density/
│   │   ├── train_wf/   # WF patches organized for EMDiffuse
│   │   └── train_gt/   # Density patches
│   └── sim2density/
│       ├── train_wf/
│       └── train_gt/
├── experiments/         # Model checkpoints & logs
└── results/             # Inference outputs
```

---

## Task 1: Environment & Cleanup

- Update `requirements.txt` for Python 3.11 + PyTorch 2.9.1 + CUDA 13.0
- Remove `3D-SR-Unet/`, `demo/`, `example/` directories
- Fix typo: `emdiffuse_conifg.py` → `emdiffuse_config.py`
- Enable cuDNN in `run.py`
- Fix EMDiffuse-r.json task field

## Task 2: Point Cloud → Image Conversion

Create modular conversion pipeline in `scripts/`:
- `scripts/utils/imaging.py` - PSF simulation, convolution, noise
- `scripts/utils/pointcloud.py` - CSV I/O, coordinate transforms
- `scripts/convert_pointcloud.py` - Main conversion entry point

## Task 3: Training Data Preparation

- Create new Dataset class supporting random cropping
- Create config files for WF→Density, SIM→Density
- Organize data into EMDiffuse-compatible format

## Task 4: Training & Inference

- Adapt training for multi-GPU RTX 5090
- Create inference pipeline
- Density map → point cloud sampling

## Task 5: Evaluation & Documentation

- Point cloud comparison metrics
- Visualization tools
- Rewrite README and documentation
