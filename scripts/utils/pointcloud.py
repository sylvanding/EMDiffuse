"""Point cloud I/O and coordinate processing utilities."""

import os
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


class PointCloudIO:
    """Read and write point cloud CSV files."""

    @staticmethod
    def read_csv(csv_path: str) -> np.ndarray:
        """Read point cloud from CSV. Returns Nx3 array in nm."""
        df = pd.read_csv(csv_path)
        expected_cols = ['x [nm]', 'y [nm]', 'z [nm]']
        if not all(c in df.columns for c in expected_cols):
            raise ValueError(f"CSV must have columns {expected_cols}, got {list(df.columns)}")
        return df[expected_cols].values.astype(np.float64)

    @staticmethod
    def write_csv(points: np.ndarray, csv_path: str):
        """Write Nx3 point cloud to CSV in nm."""
        df = pd.DataFrame(points[:, :3], columns=['x [nm]', 'y [nm]', 'z [nm]'])
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)

    @staticmethod
    def discover_datasets(root_dir: str, pattern: str = r'microtubule_(\d+)_(\d+)k') -> list:
        """Discover all point cloud folders matching the naming pattern.

        Returns list of dicts with keys: folder_path, sample_id, approx_points, csv_path
        """
        datasets = []
        for folder_name in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            match = re.match(pattern, folder_name)
            if match is None:
                continue
            csv_path = os.path.join(folder_path, 'microtubule_simulation.csv')
            if not os.path.exists(csv_path):
                continue
            datasets.append({
                'folder_name': folder_name,
                'folder_path': folder_path,
                'sample_id': int(match.group(1)),
                'approx_points': int(match.group(2)) * 1000,
                'csv_path': csv_path,
            })
        return datasets


class PointCloudProcessor:
    """Process point clouds: projection, normalization, coordinate transforms."""

    def __init__(self, image_size: int = 1024, pixel_size_nm: float = 25.0):
        self.image_size = image_size
        self.pixel_size_nm = pixel_size_nm
        self.fov_nm = image_size * pixel_size_nm  # 25600 nm = 25.6 µm

    def project_2d(self, points_3d: np.ndarray,
                   z_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """Project 3D points to 2D by dropping z (optionally filtering by z range).

        Args:
            points_3d: Nx3 array (x, y, z) in nm
            z_range: optional (z_min, z_max) to select a z-slice

        Returns:
            Nx2 array (x, y) in nm
        """
        if z_range is not None:
            z_min, z_max = z_range
            mask = (points_3d[:, 2] >= z_min) & (points_3d[:, 2] <= z_max)
            points_3d = points_3d[mask]
        return points_3d[:, :2].copy()

    def nm_to_pixel(self, points_2d_nm: np.ndarray,
                    center: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
        """Convert nm coordinates to pixel coordinates, centered in the FOV.

        Args:
            points_2d_nm: Nx2 array (x, y) in nm
            center: optional 2D center point in nm. If None, uses data centroid.

        Returns:
            points_pixel: Nx2 array (x, y) in pixel coordinates
            transform_info: dict with offset, scale info for inverse transform
        """
        if center is None:
            center = points_2d_nm.mean(axis=0)

        half_fov = self.fov_nm / 2.0
        offset = center - half_fov

        points_pixel = (points_2d_nm - offset) / self.pixel_size_nm

        transform_info = {
            'offset_nm': offset,
            'center_nm': center,
            'pixel_size_nm': self.pixel_size_nm,
            'image_size': self.image_size,
            'fov_nm': self.fov_nm,
        }
        return points_pixel, transform_info

    def pixel_to_nm(self, points_pixel: np.ndarray, transform_info: dict) -> np.ndarray:
        """Convert pixel coordinates back to nm coordinates."""
        offset = transform_info['offset_nm']
        pixel_size = transform_info['pixel_size_nm']
        return points_pixel * pixel_size + offset

    def filter_in_bounds(self, points_pixel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Filter points that fall within the image bounds.

        Returns:
            filtered_points: points within [0, image_size)
            mask: boolean mask of valid points
        """
        mask = (
            (points_pixel[:, 0] >= 0) & (points_pixel[:, 0] < self.image_size) &
            (points_pixel[:, 1] >= 0) & (points_pixel[:, 1] < self.image_size)
        )
        return points_pixel[mask], mask
