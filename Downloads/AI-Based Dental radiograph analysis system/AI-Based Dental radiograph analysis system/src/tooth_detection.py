from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np


Box = Tuple[int, int, int, int]
Center = Tuple[int, int]


def postprocess_mask(prob_mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    binary = (prob_mask >= threshold).astype(np.uint8) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    return binary


def detect_teeth_boxes(binary_mask: np.ndarray, min_area: int = 200) -> List[Box]:
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Box] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, w, h))
    return boxes


def tooth_centers(boxes: List[Box]) -> List[Center]:
    return [(x + w // 2, y + h // 2) for x, y, w, h in boxes]


def map_centers_to_boxes(centers: List[Center], boxes: List[Box]) -> Dict[Center, Box]:
    out: Dict[Center, Box] = {}
    for c, b in zip(centers, boxes):
        out[c] = b
    return out
