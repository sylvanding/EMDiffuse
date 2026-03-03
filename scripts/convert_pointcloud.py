"""Convert point cloud CSV files to microscopy images (WF/SIM/STED/density).

Usage:
    # Convert a few samples for verification:
    python scripts/convert_pointcloud.py \
        --input_dir /data0/djx/img2pc_2d/microtubules \
        --output_dir /data0/djx/EMDiffuse/images/microtubules \
        --samples 3 --visualize

    # Convert all samples:
    python scripts/convert_pointcloud.py \
        --input_dir /data0/djx/img2pc_2d/microtubules \
        --output_dir /data0/djx/EMDiffuse/images/microtubules \
        --samples all

    # Convert specific modalities only:
    python scripts/convert_pointcloud.py \
        --input_dir /data0/djx/img2pc_2d/microtubules \
        --output_dir /data0/djx/EMDiffuse/images/microtubules \
        --modalities wf density
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils.pointcloud import PointCloudIO, PointCloudProcessor
from scripts.utils.imaging import MicroscopyImageSimulator, ModalityConfig, MODALITY_PRESETS


def convert_single_sample(csv_path: str, output_dir: str, sample_id: str,
                          processor: PointCloudProcessor,
                          simulator: MicroscopyImageSimulator,
                          modalities: list, seed: int = 42,
                          visualize: bool = False) -> dict:
    """Convert a single point cloud CSV to microscopy images.

    Returns dict with metadata about the conversion.
    """
    points_3d = PointCloudIO.read_csv(csv_path)
    n_total = len(points_3d)

    points_2d = processor.project_2d(points_3d)
    points_pixel, transform_info = processor.nm_to_pixel(points_2d)
    points_pixel, valid_mask = processor.filter_in_bounds(points_pixel)
    n_valid = len(points_pixel)

    images = simulator.generate_all_modalities(points_pixel, modalities, seed=seed)

    metadata = {
        'sample_id': sample_id,
        'csv_path': csv_path,
        'n_points_total': int(n_total),
        'n_points_valid': int(n_valid),
        'n_points_dropped': int(n_total - n_valid),
        'transform_info': {k: v.tolist() if isinstance(v, np.ndarray) else v
                           for k, v in transform_info.items()},
        'modalities': {},
    }

    for mod_name, image in images.items():
        mod_dir = os.path.join(output_dir, mod_name)
        os.makedirs(mod_dir, exist_ok=True)

        tiff_path = os.path.join(mod_dir, f'{sample_id}.tif')
        img_uint16 = (image * 65535).astype(np.uint16)
        tifffile.imwrite(tiff_path, img_uint16)

        metadata['modalities'][mod_name] = {
            'path': tiff_path,
            'min': float(image.min()),
            'max': float(image.max()),
            'mean': float(image.mean()),
            'nonzero_fraction': float((image > 0.001).sum() / image.size),
        }

    if visualize:
        _visualize_sample(images, sample_id, output_dir, points_pixel, processor.image_size)

    return metadata


def _visualize_sample(images: dict, sample_id: str, output_dir: str,
                      points_pixel: np.ndarray, image_size: int):
    """Create visualization comparing all modalities."""
    n_mods = len(images)
    fig, axes = plt.subplots(1, n_mods + 1, figsize=(5 * (n_mods + 1), 5))

    ax = axes[0]
    subsample = max(1, len(points_pixel) // 10000)
    pts = points_pixel[::subsample]
    ax.scatter(pts[:, 0], pts[:, 1], s=0.1, c='blue', alpha=0.3)
    ax.set_xlim(0, image_size)
    ax.set_ylim(image_size, 0)
    ax.set_title(f'Point Cloud ({len(points_pixel):,} pts)')
    ax.set_aspect('equal')

    for i, (mod_name, image) in enumerate(images.items()):
        ax = axes[i + 1]
        ax.imshow(image, cmap='hot', vmin=0, vmax=image.max() * 0.8)
        ax.set_title(f'{mod_name.upper()} (max={image.max():.3f})')
        ax.axis('off')

    plt.suptitle(f'Sample {sample_id}', fontsize=14)
    plt.tight_layout()

    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    plt.savefig(os.path.join(vis_dir, f'{sample_id}_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Convert point cloud CSVs to microscopy images')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Root directory containing point cloud folders')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for generated images')
    parser.add_argument('--samples', type=str, default='3',
                        help='Number of samples to convert, or "all"')
    parser.add_argument('--modalities', nargs='+', default=None,
                        help=f'Modalities to generate (default: all). Options: {list(MODALITY_PRESETS.keys())}')
    parser.add_argument('--image_size', type=int, default=1024,
                        help='Output image size in pixels (default: 1024)')
    parser.add_argument('--pixel_size', type=float, default=25.0,
                        help='Pixel size in nm (default: 25.0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for noise generation')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate comparison visualizations')
    parser.add_argument('--pattern', type=str, default=r'microtubule_(\d+)_(\d+)k',
                        help='Regex pattern for folder names')
    parser.add_argument('--structure_type', type=str, default='microtubules',
                        help='Biological structure type (for metadata)')
    args = parser.parse_args()

    modalities = args.modalities or list(MODALITY_PRESETS.keys())
    for m in modalities:
        if m not in MODALITY_PRESETS:
            raise ValueError(f"Unknown modality '{m}'. Options: {list(MODALITY_PRESETS.keys())}")

    print(f"{'='*60}")
    print(f"Point Cloud to Microscopy Image Conversion")
    print(f"{'='*60}")
    print(f"Input:       {args.input_dir}")
    print(f"Output:      {args.output_dir}")
    print(f"Image size:  {args.image_size}x{args.image_size}")
    print(f"Pixel size:  {args.pixel_size} nm")
    print(f"FOV:         {args.image_size * args.pixel_size / 1000:.1f} um")
    print(f"Modalities:  {modalities}")
    print(f"Structure:   {args.structure_type}")
    print(f"{'='*60}")

    datasets = PointCloudIO.discover_datasets(args.input_dir, args.pattern)
    print(f"Found {len(datasets)} point cloud datasets")

    if args.samples != 'all':
        n_samples = int(args.samples)
        datasets = datasets[:n_samples]
        print(f"Processing first {n_samples} samples")
    else:
        print(f"Processing all {len(datasets)} samples")

    processor = PointCloudProcessor(args.image_size, args.pixel_size)
    simulator = MicroscopyImageSimulator(args.image_size, args.pixel_size)

    os.makedirs(args.output_dir, exist_ok=True)
    all_metadata = {
        'structure_type': args.structure_type,
        'image_size': args.image_size,
        'pixel_size_nm': args.pixel_size,
        'fov_nm': args.image_size * args.pixel_size,
        'modalities': modalities,
        'samples': {},
    }

    t_start = time.time()
    for i, ds in enumerate(datasets):
        sid = f"{ds['sample_id']:04d}"
        t0 = time.time()

        try:
            meta = convert_single_sample(
                csv_path=ds['csv_path'],
                output_dir=args.output_dir,
                sample_id=sid,
                processor=processor,
                simulator=simulator,
                modalities=modalities,
                seed=args.seed + ds['sample_id'],
                visualize=args.visualize,
            )
            all_metadata['samples'][sid] = meta
            dt = time.time() - t0
            print(f"[{i+1}/{len(datasets)}] {ds['folder_name']} -> {sid} "
                  f"({meta['n_points_valid']:,} pts, {dt:.1f}s)")

        except Exception as e:
            print(f"[{i+1}/{len(datasets)}] ERROR processing {ds['folder_name']}: {e}")
            import traceback
            traceback.print_exc()

    meta_path = os.path.join(args.output_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"Processed: {len(all_metadata['samples'])} samples")
    print(f"Total time: {total_time:.1f}s ({total_time/max(len(all_metadata['samples']),1):.1f}s/sample)")
    print(f"Output:     {args.output_dir}")
    print(f"Metadata:   {meta_path}")
    if args.visualize:
        print(f"Visuals:    {os.path.join(args.output_dir, 'visualizations')}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
