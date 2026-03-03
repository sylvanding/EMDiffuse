"""Dataset for 2D super-resolution: WF/SIM -> Density map.

Supports both pre-cropped patches (EMDiffuse-compatible) and on-the-fly random cropping.
"""

import os
import random

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image, ImageFilter, ImageOps
from tifffile import imread
from torchvision import transforms


class SuperResolutionDataset(data.Dataset):
    """Dataset for microscopy super-resolution (WF/SIM -> Density).

    Supports two modes:
    1. Pre-cropped patches (patch_mode=True): reads from train_wf/train_gt structure
    2. Full images with random cropping (patch_mode=False): reads full-size images

    Directory structure for patch mode:
        data_root/
            {sample_id}/
                wf/
                    0.tif, 1.tif, ...
        (GT path is derived by replacing 'wf' with 'gt' in path)

    Directory structure for full image mode:
        input_dir/
            0001.tif, 0002.tif, ...
        target_dir/
            0001.tif, 0002.tif, ...
    """

    def __init__(self, data_root, data_len=-1, norm=True, percent=False,
                 phase='train', image_size=(256, 256),
                 input_dir=None, target_dir=None,
                 patch_mode=True, random_crop=True,
                 loader=None):
        self.data_root = data_root
        self.phase = phase
        self.norm = norm
        self.image_size = image_size if isinstance(image_size, (list, tuple)) else (image_size, image_size)
        self.patch_mode = patch_mode
        self.random_crop = random_crop and (phase == 'train')

        if patch_mode:
            self.img_paths, self.gt_paths = self._read_patch_dataset(data_root)
        else:
            if input_dir is None or target_dir is None:
                raise ValueError("input_dir and target_dir required for full image mode")
            self.img_paths, self.gt_paths = self._read_full_dataset(input_dir, target_dir)

        if data_len > 0:
            self.img_paths = self.img_paths[:data_len]
            self.gt_paths = self.gt_paths[:data_len]

        self.tfs = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.tfs_notresize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __getitem__(self, index):
        ret = {}
        img_path = self.img_paths[index]
        gt_path = self.gt_paths[index]

        img = self._load_image(img_path)
        gt = self._load_image(gt_path)

        if self.random_crop and not self.patch_mode:
            img, gt = self._random_crop(img, gt)

        if self.phase == 'train':
            img, gt = self._augment(img, gt)

        if self.patch_mode:
            img = self.tfs(img)
            gt = self.tfs(gt)
        else:
            img = self.tfs(img)
            gt = self.tfs(gt)

        ret['gt_image'] = gt
        ret['cond_image'] = img
        ret['path'] = '_'.join(img_path.split(os.sep)[-3:])
        return ret

    def __len__(self):
        return len(self.img_paths)

    def _load_image(self, path: str) -> Image.Image:
        """Load image as PIL grayscale."""
        if path.endswith('.tif') or path.endswith('.tiff'):
            arr = imread(path)
            if arr.dtype == np.uint16:
                arr = (arr / 256).astype(np.uint8)
            elif arr.dtype == np.float32 or arr.dtype == np.float64:
                arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            return Image.fromarray(arr, mode='L')
        return Image.open(path).convert('L')

    def _random_crop(self, img: Image.Image, gt: Image.Image):
        """Random crop both images at the same location."""
        w, h = img.size
        th, tw = self.image_size

        if w < tw or h < th:
            img = img.resize((max(w, tw), max(h, th)), Image.BILINEAR)
            gt = gt.resize((max(w, tw), max(h, th)), Image.BILINEAR)
            w, h = img.size

        x = random.randint(0, w - tw)
        y = random.randint(0, h - th)

        img = img.crop((x, y, x + tw, y + th))
        gt = gt.crop((x, y, x + tw, y + th))
        return img, gt

    def _augment(self, img: Image.Image, gt: Image.Image):
        """Data augmentation: flip, rotate, blur."""
        if random.random() < 0.5:
            img = ImageOps.flip(img)
            gt = ImageOps.flip(gt)
        if random.random() < 0.5:
            img = ImageOps.mirror(img)
            gt = ImageOps.mirror(gt)
        if random.random() < 0.5:
            angle = random.choice([90, 180, 270])
            img = img.rotate(angle)
            gt = gt.rotate(angle)
        if random.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))
        return img, gt

    def _read_patch_dataset(self, data_root):
        """Read pre-cropped patch dataset (EMDiffuse-compatible format)."""
        img_paths = []
        gt_paths = []
        for cell_num in sorted(os.listdir(data_root)):
            if cell_num.startswith('.'):
                continue
            cell_path = os.path.join(data_root, cell_num)
            if not os.path.isdir(cell_path):
                continue
            for noise_level in sorted(os.listdir(cell_path)):
                level_path = os.path.join(cell_path, noise_level)
                if not os.path.isdir(level_path):
                    continue
                for img_name in sorted(os.listdir(level_path)):
                    if img_name.endswith(('.tif', '.tiff', '.png')):
                        img_full = os.path.join(level_path, img_name)
                        gt_full = img_full.replace('wf', 'gt')
                        if os.path.exists(gt_full):
                            img_paths.append(img_full)
                            gt_paths.append(gt_full)
        return img_paths, gt_paths

    def _read_full_dataset(self, input_dir, target_dir):
        """Read full-size image pairs."""
        img_paths = []
        gt_paths = []
        for fname in sorted(os.listdir(input_dir)):
            if not fname.endswith(('.tif', '.tiff', '.png')):
                continue
            inp_path = os.path.join(input_dir, fname)
            tgt_path = os.path.join(target_dir, fname)
            if os.path.exists(tgt_path):
                img_paths.append(inp_path)
                gt_paths.append(tgt_path)
        return img_paths, gt_paths
