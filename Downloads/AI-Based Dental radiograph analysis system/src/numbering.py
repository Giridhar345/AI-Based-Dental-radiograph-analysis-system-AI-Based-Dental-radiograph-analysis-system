from __future__ import annotations

from typing import Dict, List, Tuple


Center = Tuple[int, int]


def _split_upper_lower(centers: List[Center]) -> Tuple[List[Center], List[Center]]:
    if not centers:
        return [], []
    ys = [c[1] for c in centers]
    y_mid = sum(ys) / len(ys)
    upper = [c for c in centers if c[1] <= y_mid]
    lower = [c for c in centers if c[1] > y_mid]
    return upper, lower


def assign_fdi_numbers(centers: List[Center]) -> Dict[Center, int]:
    upper, lower = _split_upper_lower(centers)

    upper_sorted = sorted(upper, key=lambda c: c[0])
    lower_sorted = sorted(lower, key=lambda c: c[0])

    mapping: Dict[Center, int] = {}

    upper_left = [c for c in upper_sorted if c[0] <= (upper_sorted[len(upper_sorted) // 2][0] if upper_sorted else 0)]
    upper_right = [c for c in upper_sorted if c not in upper_left]

    lower_left = [c for c in lower_sorted if c[0] <= (lower_sorted[len(lower_sorted) // 2][0] if lower_sorted else 0)]
    lower_right = [c for c in lower_sorted if c not in lower_left]

    upper_right_sorted = sorted(upper_right, key=lambda c: c[0], reverse=True)
    for i, c in enumerate(upper_right_sorted[:8]):
        mapping[c] = 11 + i

    upper_left_sorted = sorted(upper_left, key=lambda c: c[0])
    for i, c in enumerate(upper_left_sorted[:8]):
        mapping[c] = 21 + i

    lower_left_sorted = sorted(lower_left, key=lambda c: c[0])
    for i, c in enumerate(lower_left_sorted[:8]):
        mapping[c] = 31 + i

    lower_right_sorted = sorted(lower_right, key=lambda c: c[0], reverse=True)
    for i, c in enumerate(lower_right_sorted[:8]):
        mapping[c] = 41 + i

    return mapping
