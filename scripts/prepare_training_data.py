"""Prepare training data for diffusion model: crop patches and organize into EMDiffuse format.

Usage:
    # Prepare WF->Density training data:
    python scripts/prepare_training_data.py \
        --image_dir /data0/djx/EMDiffuse/images/microtubules \
        --output_dir /data0/djx/EMDiffuse/training/wf2density \
        --input_modality wf \
        --target_modality density \
        --patch_size 256 --overlap 0.125 \
        --train_ratio 0.9

    # Prepare SIM->Density training data:
    python scripts/prepare_training_data.py \
        --image_dir /data0/djx/EMDiffuse/images/microtubules \
        --output_dir /data0/djx/EMDiffuse/training/sim2density \
        --input_modality sim \
        --target_modality density \
        --patch_size 256 --overlap 0.125
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import tifffile
from tqdm import tqdm


def crop_patches(image: np.ndarray, patch_size: int, overlap: float) -> list:
    """Crop an image into overlapping patches.

    Returns list of (patch, row_start, col_start) tuples.
    """
    h, w = image.shape
    stride = int(patch_size * (1 - overlap))
    patches = []

    for r in range(0, h - patch_size + 1, stride):
        for c in range(0, w - patch_size + 1, stride):
            patch = image[r:r+patch_size, c:c+patch_size]
            patches.append((patch, r, c))

    return patches


def has_content(patch: np.ndarray, threshold: float = 0.001,
                min_fraction: float = 0.01) -> bool:
    """Check if a patch has meaningful content (not just background)."""
    return (patch > threshold).sum() / patch.size >= min_fraction


def main():
    parser = argparse.ArgumentParser(description='Prepare training data for diffusion model')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing modality folders (e.g., wf/, density/)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for training data')
    parser.add_argument('--input_modality', type=str, required=True,
                        help='Input modality name (e.g., wf, sim)')
    parser.add_argument('--target_modality', type=str, default='density',
                        help='Target modality name (default: density)')
    parser.add_argument('--patch_size', type=int, default=256,
                        help='Patch size in pixels (default: 256)')
    parser.add_argument('--overlap', type=float, default=0.125,
                        help='Overlap ratio between patches (default: 0.125)')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help='Fraction of samples for training (default: 0.9)')
    parser.add_argument('--min_content', type=float, default=0.01,
                        help='Minimum content fraction to keep a patch (default: 0.01)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    input_dir = os.path.join(args.image_dir, args.input_modality)
    target_dir = os.path.join(args.image_dir, args.target_modality)

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input modality directory not found: {input_dir}")
    if not os.path.exists(target_dir):
        raise FileNotFoundError(f"Target modality directory not found: {target_dir}")

    input_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.tif')])
    print(f"Found {len(input_files)} images")

    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(input_files))
    n_train = int(len(input_files) * args.train_ratio)
    train_indices = set(indices[:n_train])

    train_wf_dir = os.path.join(args.output_dir, 'train_wf')
    train_gt_dir = os.path.join(args.output_dir, 'train_gt')
    test_wf_dir = os.path.join(args.output_dir, 'test_wf')
    test_gt_dir = os.path.join(args.output_dir, 'test_gt')

    stats = {'train_patches': 0, 'test_patches': 0, 'skipped_empty': 0}

    for file_idx, filename in enumerate(tqdm(input_files, desc='Processing')):
        sample_id = filename.replace('.tif', '')
        input_path = os.path.join(input_dir, filename)
        target_path = os.path.join(target_dir, filename)

        if not os.path.exists(target_path):
            print(f"  Warning: target not found for {filename}, skipping")
            continue

        input_img = tifffile.imread(input_path).astype(np.float32)
        target_img = tifffile.imread(target_path).astype(np.float32)

        if input_img.max() > 1.0:
            input_img = input_img / 65535.0
        if target_img.max() > 1.0:
            target_img = target_img / 65535.0

        input_patches = crop_patches(input_img, args.patch_size, args.overlap)
        target_patches = crop_patches(target_img, args.patch_size, args.overlap)

        is_train = file_idx in train_indices
        split = 'train' if is_train else 'test'

        wf_base = train_wf_dir if is_train else test_wf_dir
        gt_base = train_gt_dir if is_train else test_gt_dir

        wf_sample_dir = os.path.join(wf_base, sample_id, 'wf')
        gt_sample_dir = os.path.join(gt_base, sample_id, 'gt')
        os.makedirs(wf_sample_dir, exist_ok=True)
        os.makedirs(gt_sample_dir, exist_ok=True)

        patch_idx = 0
        for (inp_patch, _, _), (tgt_patch, _, _) in zip(input_patches, target_patches):
            if not has_content(tgt_patch, min_fraction=args.min_content):
                stats['skipped_empty'] += 1
                continue

            inp_uint8 = (np.clip(inp_patch, 0, 1) * 255).astype(np.uint8)
            tgt_uint8 = (np.clip(tgt_patch, 0, 1) * 255).astype(np.uint8)

            tifffile.imwrite(os.path.join(wf_sample_dir, f'{patch_idx}.tif'), inp_uint8)
            tifffile.imwrite(os.path.join(gt_sample_dir, f'{patch_idx}.tif'), tgt_uint8)

            if is_train:
                stats['train_patches'] += 1
            else:
                stats['test_patches'] += 1
            patch_idx += 1

    meta = {
        'input_modality': args.input_modality,
        'target_modality': args.target_modality,
        'patch_size': args.patch_size,
        'overlap': args.overlap,
        'train_ratio': args.train_ratio,
        'n_images': len(input_files),
        'n_train_images': n_train,
        'n_test_images': len(input_files) - n_train,
        **stats,
    }
    meta_path = os.path.join(args.output_dir, 'training_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Training data preparation complete!")
    print(f"  Train patches: {stats['train_patches']}")
    print(f"  Test patches:  {stats['test_patches']}")
    print(f"  Skipped empty: {stats['skipped_empty']}")
    print(f"  Output:        {args.output_dir}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
