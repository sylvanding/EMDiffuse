"""Test the full sampling pipeline: density map -> sample points -> reconstruct density -> compare.

This script validates the sampling quality by:
1. Loading a ground truth density map
2. Sampling N points from it
3. Reconstructing a density map from sampled points
4. Comparing original and reconstructed density maps visually and numerically

Usage:
    python scripts/test_sampling_pipeline.py \
        --image_dir /data0/djx/EMDiffuse/images/microtubules \
        --output_dir /data0/djx/EMDiffuse/sampling_test \
        --n_points 50000 100000 200000 400000 \
        --samples 3
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
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils.imaging import DensitySampler, MicroscopyImageSimulator, ModalityConfig
from scripts.utils.pointcloud import PointCloudProcessor


def reconstruct_density_from_points(points_pixel, image_size=1024, sigma=1.0):
    """Reconstruct density map from sampled pixel-space points."""
    hist, _, _ = np.histogram2d(
        points_pixel[:, 1], points_pixel[:, 0],
        bins=image_size, range=[[0, image_size], [0, image_size]]
    )
    density = gaussian_filter(hist.astype(np.float64), sigma=sigma)
    if density.max() > 0:
        density /= density.max()
    return density


def test_single_sample(density_path, sample_id, n_points_list,
                       output_dir, pixel_size_nm=25.0, image_size=1024,
                       transform_info=None, seed=42):
    """Test sampling at different point counts for a single density map."""
    density_gt = tifffile.imread(density_path).astype(np.float64)
    if density_gt.max() > 1:
        density_gt /= density_gt.max()

    sampler = DensitySampler(pixel_size_nm=pixel_size_nm)
    processor = PointCloudProcessor(image_size, pixel_size_nm)

    n_cols = len(n_points_list) + 1
    fig, axes = plt.subplots(3, n_cols, figsize=(5 * n_cols, 15))

    axes[0, 0].imshow(density_gt, cmap='hot')
    axes[0, 0].set_title('GT Density Map')
    axes[0, 0].axis('off')

    sim = MicroscopyImageSimulator(image_size, pixel_size_nm)
    wf_config = ModalityConfig.from_preset('wf', pixel_size_nm, noise_enabled=False)

    pts_2d_gt = _density_to_pixel_coords(density_gt, image_size)
    wf_gt = sim.simulate_modality(pts_2d_gt, wf_config, np.random.default_rng(seed))
    axes[1, 0].imshow(wf_gt, cmap='gray')
    axes[1, 0].set_title('GT -> WF')
    axes[1, 0].axis('off')

    axes[2, 0].text(0.1, 0.5, 'Ground Truth\nReference',
                    fontsize=14, ha='left', va='center',
                    transform=axes[2, 0].transAxes)
    axes[2, 0].axis('off')

    results = {}
    for i, n_pts in enumerate(n_points_list):
        rng = np.random.default_rng(seed + i)
        points = sampler.sample_points(density_gt, n_pts, transform_info=None, rng=rng)
        recon = reconstruct_density_from_points(points, image_size, sigma=1.0)

        mse = np.mean((density_gt - recon) ** 2)
        psnr = -10 * np.log10(mse + 1e-10)

        gt_c = density_gt - density_gt.mean()
        re_c = recon - recon.mean()
        pcc = np.mean(gt_c * re_c) / (gt_c.std() * re_c.std() + 1e-10)

        results[n_pts] = {'mse': float(mse), 'psnr': float(psnr), 'pcc': float(pcc)}

        axes[0, i + 1].imshow(recon, cmap='hot')
        axes[0, i + 1].set_title(f'Sampled {n_pts//1000}k pts')
        axes[0, i + 1].axis('off')

        wf_recon = sim.simulate_modality(points, wf_config, np.random.default_rng(seed))
        axes[1, i + 1].imshow(wf_recon, cmap='gray')
        axes[1, i + 1].set_title(f'{n_pts//1000}k pts -> WF')
        axes[1, i + 1].axis('off')

        diff = np.abs(density_gt - recon)
        axes[2, i + 1].imshow(diff, cmap='hot',
                               vmax=max(0.01, diff.max() * 0.5))
        metric_str = f'PSNR: {psnr:.1f} dB\nPCC: {pcc:.4f}\nMSE: {mse:.6f}'
        axes[2, i + 1].set_title(f'Diff | {metric_str}', fontsize=9)
        axes[2, i + 1].axis('off')

    plt.suptitle(f'Sampling Pipeline Test - Sample {sample_id}', fontsize=16)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, f'{sample_id}_sampling_test.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Sample {sample_id}: saved -> {fig_path}")
    for n_pts, m in results.items():
        print(f"    {n_pts//1000}k pts: PSNR={m['psnr']:.1f}dB, PCC={m['pcc']:.4f}")

    return results


def _density_to_pixel_coords(density, image_size):
    """Extract approximate pixel coordinates from a density map for WF simulation."""
    coords = np.argwhere(density > 0.01)
    if len(coords) == 0:
        return np.zeros((1, 2))
    weights = density[coords[:, 0], coords[:, 1]]
    n_repeat = np.maximum(1, (weights * 100).astype(int))
    rows = np.repeat(coords[:, 0], n_repeat)
    cols = np.repeat(coords[:, 1], n_repeat)
    jitter = np.random.uniform(-0.5, 0.5, size=(len(rows), 2))
    return np.column_stack([cols + jitter[:, 0], rows + jitter[:, 1]])


def main():
    parser = argparse.ArgumentParser(description='Test sampling pipeline quality')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory with density maps')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--n_points', type=int, nargs='+',
                        default=[50000, 100000, 200000, 400000])
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    density_dir = os.path.join(args.image_dir, 'density')
    meta_path = os.path.join(args.image_dir, 'metadata.json')

    metadata = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)

    tifs = sorted([f for f in os.listdir(density_dir) if f.endswith('.tif')])
    tifs = tifs[:args.samples]

    print(f"{'='*60}")
    print(f"Sampling Pipeline Test")
    print(f"  Density maps: {density_dir}")
    print(f"  Point counts: {[f'{n//1000}k' for n in args.n_points]}")
    print(f"  Samples: {len(tifs)}")
    print(f"{'='*60}")

    all_results = {}
    for tif_name in tifs:
        sid = tif_name.replace('.tif', '')
        density_path = os.path.join(density_dir, tif_name)

        t_info = None
        if metadata and sid in metadata.get('samples', {}):
            t_info = metadata['samples'][sid].get('transform_info')

        results = test_single_sample(
            density_path, sid, args.n_points,
            args.output_dir,
            pixel_size_nm=metadata['pixel_size_nm'] if metadata else 25.0,
            image_size=metadata['image_size'] if metadata else 1024,
            transform_info=t_info, seed=args.seed
        )
        all_results[sid] = results

    results_path = os.path.join(args.output_dir, 'sampling_test_results.json')
    serializable = {sid: {str(k): v for k, v in res.items()}
                    for sid, res in all_results.items()}
    with open(results_path, 'w') as f:
        json.dump(serializable, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print(f"Visualizations in: {args.output_dir}")


if __name__ == '__main__':
    main()
