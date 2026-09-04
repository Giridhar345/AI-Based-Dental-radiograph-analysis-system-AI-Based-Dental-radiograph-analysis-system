from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(self, images_dir: Path, masks_dir: Path, img_size: int = 512):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.img_size = img_size

        exts = ("*.png", "*.jpg", "*.jpeg")
        files: List[Path] = []
        for ext in exts:
            files.extend(sorted(self.images_dir.glob(ext)))
        self.image_files = files

    def __len__(self) -> int:
        return len(self.image_files)

    def _resolve_mask_path(self, image_path: Path) -> Path:
        stem = image_path.stem
        candidates = [
            self.masks_dir / f"{stem}.png",
            self.masks_dir / f"{stem}.jpg",
            self.masks_dir / f"{stem}.jpeg",
        ]
        for c in candidates:
            if c.exists():
                return c
        raise FileNotFoundError(f"Mask not found for image: {image_path.name}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_files[idx]
        mask_path = self._resolve_mask_path(image_path)

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0
        mask = (mask.astype(np.float32) / 255.0 > 0.5).astype(np.float32)

        image_tensor = torch.from_numpy(image).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        return image_tensor, mask_tensor
