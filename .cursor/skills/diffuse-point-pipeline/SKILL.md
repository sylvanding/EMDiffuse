---
name: diffuse-point-pipeline
description: End-to-end pipeline for diffusion-based 2D super-resolution point cloud recovery. Use when working with point cloud conversion, microscopy simulation, diffusion training, density map sampling, or evaluation. Covers WF/SIM/STED imaging, PSF simulation, training data preparation, and point cloud reconstruction.
---

# DiffusePoint Pipeline

## Architecture Overview

Real-space DDPM (no latent space) mapping blurred microscopy → probability density maps.

```
CSV (nm) → 2D project → density/WF/SIM/STED → train Diffusion → predict density → sample points
```

## Quick Reference

### 1. Convert Point Clouds

```bash
python scripts/convert_pointcloud.py \
    --input_dir /data0/djx/img2pc_2d/microtubules \
    --output_dir /data0/djx/EMDiffuse/images/microtubules \
    --samples all --visualize
```

Key classes: `PointCloudIO`, `PointCloudProcessor`, `MicroscopyImageSimulator` in `scripts/utils/`.

### 2. Prepare Training Data

```bash
python scripts/prepare_training_data.py \
    --image_dir /data0/djx/EMDiffuse/images/microtubules \
    --output_dir /data0/djx/EMDiffuse/training/wf2density \
    --input_modality wf --target_modality density \
    --patch_size 256 --overlap 0.125
```

### 3. Train

```bash
python run.py -c config/WF2Density.json -b 8 --gpu 0,1,2,3 --port 20022 \
    --path /data0/djx/EMDiffuse/training/wf2density/train_wf --lr 5e-5
```

Monitor: `tensorboard --logdir /data0/djx/EMDiffuse/experiments/ --port 6006`

### 4. Sample Points

```bash
python scripts/sample_from_density.py \
    --density_map result.tif --n_points 400000 --output sampled.csv --visualize
```

Sampling: multinomial from density + sub-pixel jitter.

### 5. Evaluate

```bash
python scripts/evaluate.py --pred_density pred.tif --gt_density gt.tif \
    --output_dir eval/ --visualize
```

Metrics: MSE, PSNR, MAE, PCC, SSIM.

## Imaging Parameters

| Modality | PSF FWHM | σ (px@25nm) | Config class |
|----------|----------|-------------|--------------|
| WF | 300nm | 5.1 | `ModalityConfig.from_preset('wf')` |
| SIM | 120nm | 2.0 | `ModalityConfig.from_preset('sim')` |
| STED | 50nm | 0.85 | `ModalityConfig.from_preset('sted')` |
| Density | 25nm | 1.0 | `ModalityConfig.from_preset('density')` |

## Adding New Biological Structures

1. Place CSVs in `{structure}_{id}_{count}k/` folders
2. Update `--pattern` regex in conversion script
3. Adjust PSF/noise in `MODALITY_PRESETS` dict if needed
4. Create config from `WF2Density.json` template

## Key Files

| File | Purpose |
|------|---------|
| `scripts/utils/imaging.py` | PSF, noise, density, sampling |
| `scripts/utils/pointcloud.py` | CSV I/O, coordinate transforms |
| `data/sr_dataset.py` | Training dataset (patch mode) |
| `models/EMDiffuse_network.py` | DDPM forward/reverse process |
| `models/EMDiffuse_model.py` | Training loop (DiReP class) |
| `core/base_model.py` | Epoch loop, checkpoint saving |

## Common Issues

- **pandas error in LogTracker**: Use dict-based tracking, not DataFrame
- **Path replacement bug**: Never `str.replace('wf','gt')` on full paths
- **cuDNN**: Must be `torch.backends.cudnn.enabled = True`
- **Checkpoint saving**: Controlled by `warmup_epochs` in `base_model.py`
