from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd


def generate_report_csv(
    output_dir: Path,
    detected_teeth: List[int],
    missing_teeth: List[int],
    impacted_teeth: List[int],
    supernumerary_count: int,
    classifications: Dict[int, str],
    anomaly_label: str,
    anomaly_score: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for tooth in sorted(detected_teeth):
        rows.append(
            {
                "Tooth(FDI)": tooth,
                "Status": classifications.get(tooth, "Unknown"),
                "AnomalyLabel": anomaly_label,
                "AnomalyScore": anomaly_score,
            }
        )

    if not rows:
        rows = [
            {
                "Tooth(FDI)": "None",
                "Status": "No teeth detected",
                "AnomalyLabel": anomaly_label,
                "AnomalyScore": anomaly_score,
            }
        ]

    df = pd.DataFrame(rows)
    summary = {
        "DetectedCount": len(detected_teeth),
        "MissingTeeth": ",".join(map(str, missing_teeth)) if missing_teeth else "None",
        "ImpactedCandidates": ",".join(map(str, impacted_teeth)) if impacted_teeth else "None",
        "SupernumeraryCount": supernumerary_count,
    }

    for k, v in summary.items():
        df[k] = v

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"dental_report_{timestamp}.csv"
    df.to_csv(output_path, index=False)
    return output_path
