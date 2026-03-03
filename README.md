# DiffusePoint: Diffusion-based 2D Super-Resolution for Point Cloud Recovery

A diffusion-based framework for converting low-resolution microscopy images (WF/SIM) into high-resolution probability density maps, enabling precise point cloud reconstruction from blurred inputs.

## Principle

### Real-Space Diffusion Super-Resolution

This project uses a **Denoising Diffusion Probabilistic Model (DDPM)** operating entirely in **real image space** (no latent encoding) to learn the mapping from low-resolution microscopy images to high-resolution probability density maps.

**Forward Process:** During training, Gaussian noise is progressively added to the ground truth density map $y_0$ over $T$ timesteps:

$$y_t = \sqrt{\bar{\alpha}_t} \cdot y_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

**Reverse Process:** A UNet $\epsilon_\theta$ is trained to predict the noise $\epsilon$ given the noisy image $y_t$, the conditioning input (WF/SIM image), and the noise level $\bar{\alpha}_t$:

$$\hat{\epsilon} = \epsilon_\theta([y_{cond}, y_t], \bar{\alpha}_t)$$

**Inference:** Starting from pure Gaussian noise, the model iteratively denoises to produce a clean density map, conditioned on the input microscopy image. Each reverse step:

$$y_{t-1} = \mu_\theta(y_t, t) + \sigma_t \cdot z, \quad z \sim \mathcal{N}(0, I)$$

### Pipeline Overview

```
Point Cloud (CSV, nm coordinates)
    |
    v
2D Projection (xy plane)
    |
    v
Simulated Microscopy Images:
    - WF  (Widefield, PSF FWHM = 300nm)
    - SIM (Structured Illumination, PSF FWHM = 120nm)
    - STED (Stimulated Emission Depletion, PSF FWHM = 50nm)
    - Density Map (Ground truth, sigma = 25nm)
    |
    v
Diffusion Model Training:
    - WF  -> Density Map
    - SIM -> Density Map
    |
    v
Inference: Input Image -> Predicted Density Map
    |
    v
Point Cloud Sampling (multinomial from density)
    |
    v
Evaluation (compare with ground truth)
```

### Key Design Choices

- **Real-space diffusion**: No VAE or latent encoding. Diffusion operates directly on pixel values. This preserves fine structural details (e.g., individual microtubule filaments) that would be lost in latent space compression.
- **Conditional generation**: The UNet takes 2 input channels — the conditioning image (WF/SIM) and the current noisy estimate — enabling the model to use structural information from the input.
- **Probability density output**: The model outputs a normalized density map from which arbitrary numbers of points can be sampled via multinomial sampling with sub-pixel jittering.

## Environment Setup

Tested on Ubuntu with NVIDIA RTX 5090 GPUs, Python 3.11, CUDA 13.0.

```bash
conda create -n emdiffuse python=3.11 -y
conda activate emdiffuse
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

## Quick Start

### Step 1: Convert Point Clouds to Images

Convert point cloud CSVs to simulated microscopy images (WF/SIM/STED) and ground truth density maps.

```bash
# Test with 3 samples first (with visualization):
python scripts/convert_pointcloud.py \
    --input_dir /data0/djx/img2pc_2d/microtubules \
    --output_dir /data0/djx/EMDiffuse/images/microtubules \
    --samples 3 --visualize

# Convert all 1024 samples:
python scripts/convert_pointcloud.py \
    --input_dir /data0/djx/img2pc_2d/microtubules \
    --output_dir /data0/djx/EMDiffuse/images/microtubules \
    --samples all
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--image_size` | 1024 | Output image dimensions (pixels) |
| `--pixel_size` | 25.0 | Pixel size in nm |
| `--modalities` | all | Which modalities to generate (wf/sim/sted/density) |
| `--structure_type` | microtubules | Biological structure label |

### Step 2: Prepare Training Data

Crop image pairs into patches for diffusion model training.

```bash
# WF -> Density training set:
python scripts/prepare_training_data.py \
    --image_dir /data0/djx/EMDiffuse/images/microtubules \
    --output_dir /data0/djx/EMDiffuse/training/wf2density \
    --input_modality wf --target_modality density \
    --patch_size 256 --overlap 0.125 --train_ratio 0.9

# SIM -> Density training set:
python scripts/prepare_training_data.py \
    --image_dir /data0/djx/EMDiffuse/images/microtubules \
    --output_dir /data0/djx/EMDiffuse/training/sim2density \
    --input_modality sim --target_modality density \
    --patch_size 256 --overlap 0.125 --train_ratio 0.9
```

### Step 3: Train Diffusion Model

```bash
# Train WF -> Density model (multi-GPU):
python run.py -c config/WF2Density.json -b 8 --gpu 0,1,2,3 --port 20022 \
    --path /data0/djx/EMDiffuse/training/wf2density/train_wf --lr 5e-5

# Train SIM -> Density model:
python run.py -c config/SIM2Density.json -b 8 --gpu 0,1,2,3 --port 20023 \
    --path /data0/djx/EMDiffuse/training/sim2density/train_wf --lr 5e-5
```

Training checkpoints and logs are saved to `/data0/djx/EMDiffuse/experiments/`.

### Step 4: Inference

```bash
python run.py -p test -c config/WF2Density.json --gpu 0 -b 8 \
    --path /data0/djx/EMDiffuse/training/wf2density/test_wf \
    --resume /data0/djx/EMDiffuse/experiments/WF2Density/best \
    --mean 1 --step 1000
```

### Step 5: Sample Points from Density Maps

```bash
# Sample a specific number of points from predicted density:
python scripts/sample_from_density.py \
    --density_map /data0/djx/EMDiffuse/results/density/0001.tif \
    --n_points 50000 \
    --output /data0/djx/EMDiffuse/results/sampled/0001.csv \
    --metadata /data0/djx/EMDiffuse/images/microtubules/metadata.json \
    --sample_id 0001 --visualize
```

### Step 6: Evaluate Results

```bash
# Compare predicted density with ground truth:
python scripts/evaluate.py \
    --pred_density /data0/djx/EMDiffuse/results/density/0001.tif \
    --gt_density /data0/djx/EMDiffuse/images/microtubules/density/0001.tif \
    --output_dir /data0/djx/EMDiffuse/results/evaluation \
    --visualize
```

## Project Structure

```
EMDiffuse/
├── config/                          # Model configurations
│   ├── WF2Density.json              # Widefield -> Density
│   ├── SIM2Density.json             # SIM -> Density
│   ├── EMDiffuse-n.json             # (legacy) EM denoising
│   └── EMDiffuse-r.json             # (legacy) EM super-resolution
├── scripts/                         # Data processing pipeline
│   ├── convert_pointcloud.py        # Point cloud -> images
│   ├── prepare_training_data.py     # Crop patches for training
│   ├── sample_from_density.py       # Density map -> point cloud
│   ├── evaluate.py                  # Quality evaluation
│   └── utils/
│       ├── imaging.py               # PSF simulation, noise, density
│       └── pointcloud.py            # Point cloud I/O, transforms
├── models/                          # Diffusion model
│   ├── EMDiffuse_model.py           # Training/inference logic (DiReP)
│   ├── EMDiffuse_network.py         # DDPM network (forward/reverse)
│   └── guided_diffusion_modules/
│       └── unet.py                  # UNet backbone
├── data/
│   ├── dataset.py                   # Original dataset classes
│   └── sr_dataset.py                # Super-resolution dataset
├── core/                            # Training infrastructure
│   ├── base_model.py                # Training loop
│   ├── base_network.py              # Weight initialization
│   ├── praser.py                    # Config parser
│   └── logger.py                    # Logging utilities
├── run.py                           # Main entry point
├── requirements.txt
└── docs/plans/                      # Implementation plans
```

## Imaging Parameters

| Modality | PSF FWHM (nm) | σ (pixels @ 25nm/px) | Description |
|----------|:------------:|:-------------------:|-------------|
| WF | 300 | 5.1 | Widefield — diffraction limited |
| SIM | 120 | 2.0 | Structured Illumination — ~2x beyond diffraction |
| STED | 50 | 0.85 | Stimulated Emission Depletion — sub-diffraction |
| Density | 25 | 1.0 | Ground truth probability density |

## Extensibility

The pipeline is designed to support different biological structures. To add a new structure type:

1. Place point cloud CSVs in a folder with pattern `{structure}_{id}_{count}k/`
2. Set `--structure_type` and `--pattern` in the conversion script
3. Adjust PSF/noise parameters in `scripts/utils/imaging.py` if needed
4. Create new config files based on `WF2Density.json`

Supported / planned structures:
- Microtubules
- Mitochondria (planned)
- Endoplasmic reticulum (planned)

## Data Format

### Input: Point Cloud CSV
```
x [nm],y [nm],z [nm]
22731.510,10357.466,55.271
15153.398,836.224,1205.677
...
```

### Output: TIFF Images
- 16-bit unsigned integer (0-65535)
- 1024×1024 pixels at 25 nm/pixel
- FOV: 25.6 µm × 25.6 µm

## Acknowledgments

Based on the [EMDiffuse](https://github.com/Luchixiang/EMDiffuse) framework (Nature Communications, 2024). Modified for 2D super-resolution microscopy and point cloud recovery.
