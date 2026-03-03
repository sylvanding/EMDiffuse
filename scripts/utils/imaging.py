"""Microscopy image simulation: PSF convolution, noise, density map generation."""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class PSFParams:
    """Point Spread Function parameters for different imaging modalities."""
    name: str
    fwhm_nm: float
    pixel_size_nm: float = 25.0

    @property
    def sigma_nm(self) -> float:
        return self.fwhm_nm / 2.3548  # FWHM = 2*sqrt(2*ln2)*sigma

    @property
    def sigma_pixels(self) -> float:
        return self.sigma_nm / self.pixel_size_nm


# Pre-defined modality configurations (pixel_size_nm=25)
MODALITY_PRESETS = {
    'wf':      {'fwhm_nm': 300.0, 'description': 'Widefield microscopy'},
    'sim':     {'fwhm_nm': 120.0, 'description': 'Structured Illumination Microscopy'},
    'sted':    {'fwhm_nm': 50.0,  'description': 'Stimulated Emission Depletion'},
    'density': {'fwhm_nm': 25.0,  'description': 'Probability density (1px kernel)'},
}


@dataclass
class NoiseParams:
    """Noise parameters for image simulation."""
    poisson_scale: float = 1.0       # scale for Poisson noise (photon count scaling)
    gaussian_std: float = 0.02       # additive Gaussian noise std (relative to max)
    background: float = 0.01         # constant background level (relative to max)
    enabled: bool = True


@dataclass
class ModalityConfig:
    """Complete configuration for a microscopy modality."""
    name: str
    psf: PSFParams
    noise: NoiseParams = field(default_factory=NoiseParams)

    @classmethod
    def from_preset(cls, modality: str, pixel_size_nm: float = 25.0,
                    noise_enabled: bool = True) -> 'ModalityConfig':
        preset = MODALITY_PRESETS[modality]
        psf = PSFParams(name=modality, fwhm_nm=preset['fwhm_nm'], pixel_size_nm=pixel_size_nm)

        if modality == 'density':
            noise = NoiseParams(enabled=False)
        elif modality == 'wf':
            noise = NoiseParams(
                poisson_scale=0.8, gaussian_std=0.03,
                background=0.02, enabled=noise_enabled
            )
        elif modality == 'sim':
            noise = NoiseParams(
                poisson_scale=0.9, gaussian_std=0.02,
                background=0.01, enabled=noise_enabled
            )
        elif modality == 'sted':
            noise = NoiseParams(
                poisson_scale=0.95, gaussian_std=0.015,
                background=0.005, enabled=noise_enabled
            )
        else:
            noise = NoiseParams(enabled=noise_enabled)

        return cls(name=modality, psf=psf, noise=noise)


class MicroscopyImageSimulator:
    """Simulate microscopy images from 2D point coordinates."""

    def __init__(self, image_size: int = 1024, pixel_size_nm: float = 25.0):
        self.image_size = image_size
        self.pixel_size_nm = pixel_size_nm

    def points_to_histogram(self, points_pixel: np.ndarray) -> np.ndarray:
        """Create a 2D histogram (point accumulation) from pixel coordinates.

        Args:
            points_pixel: Nx2 array of (x, y) in pixel coordinates

        Returns:
            2D array of shape (image_size, image_size) with point counts per pixel
        """
        histogram, _, _ = np.histogram2d(
            points_pixel[:, 1],  # y -> row
            points_pixel[:, 0],  # x -> col
            bins=self.image_size,
            range=[[0, self.image_size], [0, self.image_size]]
        )
        return histogram.astype(np.float64)

    def generate_density_map(self, points_pixel: np.ndarray,
                             sigma_pixels: float = 1.0) -> np.ndarray:
        """Generate probability density map from point coordinates.

        The density map is a smoothed histogram normalized to [0, 1].

        Args:
            points_pixel: Nx2 array of (x, y) in pixel coordinates
            sigma_pixels: Gaussian smoothing sigma in pixels

        Returns:
            Normalized density map of shape (image_size, image_size) in [0, 1]
        """
        hist = self.points_to_histogram(points_pixel)
        density = gaussian_filter(hist, sigma=sigma_pixels)

        max_val = density.max()
        if max_val > 0:
            density = density / max_val
        return density

    def simulate_modality(self, points_pixel: np.ndarray,
                          config: ModalityConfig,
                          rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Simulate a microscopy image for a given modality.

        Args:
            points_pixel: Nx2 array of (x, y) in pixel coordinates
            config: ModalityConfig with PSF and noise parameters
            rng: random number generator for reproducibility

        Returns:
            Simulated image of shape (image_size, image_size) in [0, 1]
        """
        if rng is None:
            rng = np.random.default_rng()

        hist = self.points_to_histogram(points_pixel)

        blurred = gaussian_filter(hist, sigma=config.psf.sigma_pixels)

        max_val = blurred.max()
        if max_val > 0:
            image = blurred / max_val
        else:
            image = blurred

        if config.noise.enabled:
            image = self._add_noise(image, config.noise, rng)

        image = np.clip(image, 0.0, 1.0)
        return image

    def _add_noise(self, image: np.ndarray, noise: NoiseParams,
                   rng: np.random.Generator) -> np.ndarray:
        """Add realistic microscopy noise to an image."""
        noisy = image + noise.background

        if noise.poisson_scale > 0:
            photon_count = noisy * 1000 * noise.poisson_scale
            photon_count = np.maximum(photon_count, 0)
            noisy = rng.poisson(photon_count).astype(np.float64)
            noisy = noisy / (1000 * noise.poisson_scale)

        if noise.gaussian_std > 0:
            noisy = noisy + rng.normal(0, noise.gaussian_std, size=image.shape)

        return noisy

    def generate_all_modalities(self, points_pixel: np.ndarray,
                                modalities: list = None,
                                seed: int = 42) -> dict:
        """Generate images for all requested modalities.

        Args:
            points_pixel: Nx2 array of (x, y) in pixel coordinates
            modalities: list of modality names (default: all presets)
            seed: random seed

        Returns:
            Dict mapping modality name to image array
        """
        if modalities is None:
            modalities = list(MODALITY_PRESETS.keys())

        rng = np.random.default_rng(seed)
        results = {}

        for mod_name in modalities:
            config = ModalityConfig.from_preset(mod_name, self.pixel_size_nm)
            if mod_name == 'density':
                results[mod_name] = self.generate_density_map(
                    points_pixel, sigma_pixels=config.psf.sigma_pixels
                )
            else:
                results[mod_name] = self.simulate_modality(points_pixel, config, rng)

        return results


class DensitySampler:
    """Sample point clouds from probability density maps."""

    def __init__(self, pixel_size_nm: float = 25.0):
        self.pixel_size_nm = pixel_size_nm

    def sample_points(self, density_map: np.ndarray, n_points: int,
                      transform_info: dict = None,
                      rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample point coordinates from a density map.

        Args:
            density_map: 2D density map (any non-negative values)
            n_points: number of points to sample
            transform_info: coordinate transform info for converting back to nm
            rng: random number generator

        Returns:
            Nx2 array of sampled points (in nm if transform_info provided, else pixels)
        """
        if rng is None:
            rng = np.random.default_rng()

        flat = density_map.ravel().astype(np.float64)
        flat = np.maximum(flat, 0)
        total = flat.sum()
        if total <= 0:
            raise ValueError("Density map is all zeros, cannot sample")
        probs = flat / total

        indices = rng.choice(len(probs), size=n_points, replace=True, p=probs)

        h, w = density_map.shape
        rows = indices // w
        cols = indices % w

        # Sub-pixel jitter within each pixel
        row_jitter = rng.uniform(0, 1, size=n_points)
        col_jitter = rng.uniform(0, 1, size=n_points)

        y_pixel = rows.astype(np.float64) + row_jitter
        x_pixel = cols.astype(np.float64) + col_jitter

        points_pixel = np.column_stack([x_pixel, y_pixel])

        if transform_info is not None:
            from .pointcloud import PointCloudProcessor
            proc = PointCloudProcessor(
                image_size=transform_info['image_size'],
                pixel_size_nm=transform_info['pixel_size_nm']
            )
            points_nm = proc.pixel_to_nm(points_pixel, transform_info)
            z_col = np.zeros((n_points, 1))
            return np.column_stack([points_nm, z_col])

        return points_pixel
