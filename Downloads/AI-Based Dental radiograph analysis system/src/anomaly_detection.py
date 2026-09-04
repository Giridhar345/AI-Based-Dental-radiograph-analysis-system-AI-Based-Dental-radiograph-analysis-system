from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class AnomalyResult:
    label: str
    score: float


def detect_structural_anomaly(image_bgr: np.ndarray) -> AnomalyResult:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    dark_ratio = float((gray < 50).sum()) / float(gray.size)
    edge_ratio = float((edges > 0).sum()) / float(edges.size)
    score = 0.6 * dark_ratio + 0.4 * edge_ratio

    if score > 0.10:
        label = "Possible Peri-implantitis / Bone Loss"
    else:
        label = "No Significant Structural Anomaly"

    return AnomalyResult(label=label, score=round(score, 4))
