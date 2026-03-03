"""Evaluate super-resolution results: compare predicted density / sampled points with ground truth.

Usage:
    # Compare predicted density map with ground truth:
    python scripts/evaluate.py \
        --pred_density /data0/djx/EMDiffuse/results/density/0001.tif \
        --gt_density /data0/djx/EMDiffuse/images/microtubules/density/0001.tif \
        --gt_csv /data0/djx/img2pc_2d/microtubules/microtubule_001_490k/microtubule_simulation.csv \
        --sampled_csv /data0/djx/EMDiffuse/results/sampled/0001.csv \
        --output_dir /data0/djx/EMDiffuse/results/evaluation \
        --visualize

    # Batch evaluation:
    python scripts/evaluate.py \
        --pred_dir /data0/djx/EMDiffuse/results/density/ \
        --gt_image_dir /data0/djx/EMDiffuse/images/microtubules/density/ \
        --output_dir /data0/djx/EMDiffuse/results/evaluation
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
from scripts.utils.pointcloud import PointCloudIO, PointCloudProcessor
from scripts.utils.imaging import MicroscopyImageSimulator, ModalityConfig


def compute_image_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Compute image quality metrics between prediction and ground truth."""
    pred_f = pred.astype(np.float64)
    gt_f = gt.astype(np.float64)

    if pred_f.max() > 1:
        pred_f /= pred_f.max()
    if gt_f.max() > 1:
        gt_f /= gt_f.max()

    mse = np.mean((pred_f - gt_f) ** 2)
    psnr = -10 * np.log10(mse + 1e-10)
    mae = np.mean(np.abs(pred_f - gt_f))

    pred_centered = pred_f - pred_f.mean()
    gt_centered = gt_f - gt_f.mean()
    pred_std = pred_centered.std()
    gt_std = gt_centered.std()
    if pred_std > 0 and gt_std > 0:
        pcc = np.mean(pred_centered * gt_centered) / (pred_std * gt_std)
    else:
        pcc = 0.0

    from skimage.metrics import structural_similarity
    ssim = structural_similarity(pred_f, gt_f, data_range=1.0)

    return {
        'mse': float(mse),
        'psnr': float(psnr),
        'mae': float(mae),
        'pcc': float(pcc),
        'ssim': float(ssim),
    }


def points_to_image_for_comparison(points_csv: str, image_size: int = 1024,
                                   pixel_size_nm: float = 25.0,
                                   transform_info: dict = None) -> dict:
    """Convert point cloud CSV to WF/SIM/STED images for visual comparison."""
    points_3d = PointCloudIO.read_csv(points_csv)
    processor = PointCloudProcessor(image_size, pixel_size_nm)
    simulator = MicroscopyImageSimulator(image_size, pixel_size_nm)

    points_2d = processor.project_2d(points_3d)

    if transform_info:
        center = np.array(transform_info.get('center_nm', points_2d.mean(axis=0)))
    else:
        center = None
    points_pixel, t_info = processor.nm_to_pixel(points_2d, center=center)
    points_pixel, _ = processor.filter_in_bounds(points_pixel)

    return simulator.generate_all_modalities(points_pixel, ['wf', 'sim', 'sted'])


def visualize_evaluation(pred_density: np.ndarray, gt_density: np.ndarray,
                         metrics: dict, output_path: str,
                         pred_images: dict = None, gt_images: dict = None):
    """Create comprehensive evaluation visualization."""
    n_cols = 3
    if pred_images and gt_images:
        n_cols += len(pred_images)

    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10))

    axes[0, 0].imshow(gt_density, cmap='hot')
    axes[0, 0].set_title('GT Density')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(pred_density, cmap='hot')
    axes[0, 1].set_title('Predicted Density')
    axes[0, 1].axis('off')

    diff = np.abs(gt_density.astype(float) - pred_density.astype(float))
    if diff.max() > 0:
        diff /= diff.max()
    axes[0, 2].imshow(diff, cmap='hot')
    axes[0, 2].set_title('Abs Difference')
    axes[0, 2].axis('off')

    metric_text = '\n'.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
    axes[1, 0].text(0.1, 0.5, metric_text, fontsize=12, family='monospace',
                    verticalalignment='center', transform=axes[1, 0].transAxes)
    axes[1, 0].set_title('Metrics')
    axes[1, 0].axis('off')

    col_idx = 1
    if gt_images and pred_images:
        for mod_name in gt_images:
            if mod_name in pred_images:
                axes[1, col_idx].imshow(gt_images[mod_name], cmap='gray')
                axes[1, col_idx].set_title(f'GT {mod_name.upper()}')
                axes[1, col_idx].axis('off')

                if col_idx < n_cols - 1:
                    col_idx += 1

    for i in range(col_idx, n_cols):
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate super-resolution results')
    parser.add_argument('--pred_density', type=str, default=None)
    parser.add_argument('--gt_density', type=str, default=None)
    parser.add_argument('--gt_csv', type=str, default=None)
    parser.add_argument('--sampled_csv', type=str, default=None)
    parser.add_argument('--pred_dir', type=str, default=None)
    parser.add_argument('--gt_image_dir', type=str, default=None)
    parser.add_argument('--metadata', type=str, default=None)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.pred_density and args.gt_density:
        pred = tifffile.imread(args.pred_density).astype(np.float64)
        gt = tifffile.imread(args.gt_density).astype(np.float64)
        if pred.max() > 1:
            pred /= pred.max()
        if gt.max() > 1:
            gt /= gt.max()

        metrics = compute_image_metrics(pred, gt)
        print(f"Image Quality Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        if args.visualize:
            vis_path = os.path.join(args.output_dir, 'evaluation.png')
            visualize_evaluation(pred, gt, metrics, vis_path)
            print(f"Visualization: {vis_path}")

        results = {'image_metrics': metrics}
        with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=2)

    elif args.pred_dir and args.gt_image_dir:
        pred_files = sorted([f for f in os.listdir(args.pred_dir) if f.endswith('.tif')])
        all_metrics = {}

        for fname in pred_files:
            gt_path = os.path.join(args.gt_image_dir, fname)
            if not os.path.exists(gt_path):
                continue

            pred = tifffile.imread(os.path.join(args.pred_dir, fname)).astype(np.float64)
            gt = tifffile.imread(gt_path).astype(np.float64)
            if pred.max() > 1:
                pred /= pred.max()
            if gt.max() > 1:
                gt /= gt.max()

            sid = fname.replace('.tif', '')
            metrics = compute_image_metrics(pred, gt)
            all_metrics[sid] = metrics

        avg_metrics = {}
        if all_metrics:
            keys = list(next(iter(all_metrics.values())).keys())
            for k in keys:
                vals = [m[k] for m in all_metrics.values()]
                avg_metrics[k] = float(np.mean(vals))
                avg_metrics[f'{k}_std'] = float(np.std(vals))

        print(f"\nAverage metrics over {len(all_metrics)} samples:")
        for k, v in avg_metrics.items():
            if not k.endswith('_std'):
                std = avg_metrics.get(f'{k}_std', 0)
                print(f"  {k}: {v:.4f} +/- {std:.4f}")

        with open(os.path.join(args.output_dir, 'batch_evaluation.json'), 'w') as f:
            json.dump({'per_sample': all_metrics, 'average': avg_metrics}, f, indent=2)

    else:
        parser.error("Must specify (--pred_density + --gt_density) or (--pred_dir + --gt_image_dir)")


if __name__ == '__main__':
    main()
