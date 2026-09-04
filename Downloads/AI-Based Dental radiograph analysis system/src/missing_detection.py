from __future__ import annotations

from typing import Dict, List, Tuple

from src.config import FDI_EXPECTED


Center = Tuple[int, int]


def detect_missing_teeth(fdi_map: Dict[Center, int]) -> List[int]:
    detected = set(fdi_map.values())
    missing = sorted([n for n in FDI_EXPECTED if n not in detected])
    return missing


def detect_supernumerary_teeth(fdi_map: Dict[Center, int]) -> int:
    count = len(fdi_map)
    expected_max = 32
    return max(0, count - expected_max)


def detect_impacted_candidates(fdi_map: Dict[Center, int], y_threshold_ratio: float = 0.2) -> List[int]:
    if not fdi_map:
        return []

    centers = list(fdi_map.keys())
    ys = [c[1] for c in centers]
    y_min, y_max = min(ys), max(ys)
    span = max(1, y_max - y_min)
    threshold = span * y_threshold_ratio

    impacted = []
    for center, fdi in fdi_map.items():
        if fdi in (18, 28, 38, 48):
            if abs(center[1] - (y_min if fdi in (18, 28) else y_max)) > threshold:
                impacted.append(fdi)
    return sorted(impacted)
