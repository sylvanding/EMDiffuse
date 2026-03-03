"""Sample point clouds from probability density maps.

Usage:
    # Sample from a single density map:
    python scripts/sample_from_density.py \
        --density_map /data0/djx/EMDiffuse/results/predicted_density.tif \
        --n_points 50000 \
        --output /data0/djx/EMDiffuse/results/sampled_points.csv \
        --metadata /data0/djx/EMDiffuse/images/microtubules/metadata.json \
        --sample_id 0001

    # Batch sample from all density maps in a folder:
    python scripts/sample_from_density.py \
        --density_dir /data0/djx/EMDiffuse/results/density/ \
        --n_points 50000 \
        --output_dir /data0/djx/EMDiffuse/results/sampled_points/ \
        --metadata /data0/djx/EMDiffuse/images/microtubules/metadata.json

    # Sample with visualization:
    python scripts/sample_from_density.py \
        --density_map /data0/djx/EMDiffuse/results/predicted_density.tif \
        --n_points 50000 \
        --output /data0/djx/EMDiffuse/results/sampled_points.csv \
        --visualize
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils.imaging import DensitySampler
from scripts.utils.pointcloud import PointCloudIO, PointCloudProcessor


def sample_single(density_path: str, n_points: int, output_path: str,
                  transform_info: dict = None, seed: int = 42,
                  visualize: bool = False) -> np.ndarray:
    """Sample points from a single density map."""
    density = tifffile.imread(density_path).astype(np.float64)
    if density.max() > 1.0:
        density = density / density.max()

    sampler = DensitySampler(
        pixel_size_nm=transform_info['pixel_size_nm'] if transform_info else 25.0
    )
    rng = np.random.default_rng(seed)

    points = sampler.sample_points(density, n_points, transform_info, rng)

    PointCloudIO.write_csv(points, output_path)
    print(f"  Sampled {n_points} points -> {output_path}")

    if visualize:
        _visualize_sampling(density, points, output_path, transform_info)

    return points


def _visualize_sampling(density: np.ndarray, points: np.ndarray,
                        output_path: str, transform_info: dict = None):
    """Visualize the density map and sampled points side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(density, cmap='hot')
    axes[0].set_title('Input Density Map')
    axes[0].axis('off')

    if transform_info is not None:
        proc = PointCloudProcessor(
            transform_info['image_size'], transform_info['pixel_size_nm']
        )
        pts_px, _ = proc.nm_to_pixel(points[:, :2],
                                      center=np.array(transform_info['center_nm']))
    else:
        pts_px = points[:, :2]

    subsample = max(1, len(pts_px) // 20000)
    pts = pts_px[::subsample]

    canvas = np.zeros_like(density)
    h, w = canvas.shape
    for x, y in pts:
        ix, iy = int(x), int(y)
        if 0 <= ix < w and 0 <= iy < h:
            canvas[iy, ix] += 1
    from scipy.ndimage import gaussian_filter
    canvas = gaussian_filter(canvas, sigma=1.0)
    if canvas.max() > 0:
        canvas /= canvas.max()

    axes[1].imshow(canvas, cmap='hot')
    axes[1].set_title(f'Sampled Density ({len(points):,} pts)')
    axes[1].axis('off')

    diff = np.abs(density - canvas)
    axes[2].imshow(diff, cmap='hot')
    axes[2].set_title('Absolute Difference')
    axes[2].axis('off')

    plt.tight_layout()
    vis_path = output_path.replace('.csv', '_sampling_vis.png')
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Visualization -> {vis_path}")


def main():
    parser = argparse.ArgumentParser(description='Sample point clouds from density maps')
    parser.add_argument('--density_map', type=str, default=None,
                        help='Path to a single density map TIFF')
    parser.add_argument('--density_dir', type=str, default=None,
                        help='Directory of density map TIFFs (batch mode)')
    parser.add_argument('--n_points', type=int, required=True,
                        help='Number of points to sample')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path (single mode)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for CSVs (batch mode)')
    parser.add_argument('--metadata', type=str, default=None,
                        help='Path to metadata.json for coordinate transforms')
    parser.add_argument('--sample_id', type=str, default=None,
                        help='Sample ID for loading transform from metadata')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    transform_info = None
    metadata = None
    if args.metadata and os.path.exists(args.metadata):
        with open(args.metadata) as f:
            metadata = json.load(f)

    if args.density_map:
        if args.sample_id and metadata:
            sample_meta = metadata['samples'].get(args.sample_id, {})
            transform_info = sample_meta.get('transform_info')

        if transform_info is None:
            transform_info = {
                'pixel_size_nm': metadata['pixel_size_nm'] if metadata else 25.0,
                'image_size': metadata['image_size'] if metadata else 1024,
                'offset_nm': np.array([0.0, 0.0]),
                'center_nm': np.array([0.0, 0.0]),
                'fov_nm': (metadata['image_size'] if metadata else 1024) * (metadata['pixel_size_nm'] if metadata else 25.0),
            }

        output = args.output or args.density_map.replace('.tif', '_sampled.csv')
        sample_single(args.density_map, args.n_points, output,
                       transform_info, args.seed, args.visualize)

    elif args.density_dir:
        output_dir = args.output_dir or os.path.join(args.density_dir, 'sampled')
        os.makedirs(output_dir, exist_ok=True)

        tifs = sorted([f for f in os.listdir(args.density_dir) if f.endswith('.tif')])
        print(f"Found {len(tifs)} density maps in {args.density_dir}")

        for tif_name in tifs:
            sid = tif_name.replace('.tif', '')
            density_path = os.path.join(args.density_dir, tif_name)
            output_path = os.path.join(output_dir, f'{sid}.csv')

            t_info = None
            if metadata and sid in metadata.get('samples', {}):
                t_info = metadata['samples'][sid].get('transform_info')

            print(f"Sampling {sid}...")
            sample_single(density_path, args.n_points, output_path,
                          t_info, args.seed, args.visualize)
    else:
        parser.error("Must specify either --density_map or --density_dir")


if __name__ == '__main__':
    main()
