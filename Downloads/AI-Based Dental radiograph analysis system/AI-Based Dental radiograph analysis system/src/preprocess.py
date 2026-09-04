from __future__ import annotations

import cv2
import numpy as np
import torch


def preprocess_grayscale_image(image: np.ndarray, img_size: int = 512) -> torch.Tensor:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)
    norm = resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0)
    return tensor


def preprocess_rgb_for_classifier(crop_bgr: np.ndarray, img_size: int = 224) -> torch.Tensor:
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
    rgb = rgb.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std

    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    return tensor
