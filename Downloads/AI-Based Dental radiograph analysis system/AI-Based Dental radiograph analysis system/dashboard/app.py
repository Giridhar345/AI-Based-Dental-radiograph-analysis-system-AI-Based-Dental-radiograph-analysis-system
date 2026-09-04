from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.config import OUTPUTS_DIR
from src.inference import run_pipeline
from utils.report_generator import generate_report_csv


def draw_overlay(image_bgr: np.ndarray, boxes, fdi_map):
    out = image_bgr.copy()
    for (cx, cy), fdi in fdi_map.items():
        cv2.circle(out, (cx, cy), 3, (0, 255, 255), -1)
        cv2.putText(out, str(fdi), (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    for x, y, w, h in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 128, 0), 1)
    return out


st.set_page_config(page_title="Use Case-2 Dental AI", layout="wide")
st.title("Use Case-2: AI-Based Dental Radiograph Analysis")
st.caption("OPG segmentation, FDI numbering, missing/impacted/supernumerary checks, classification, anomaly detection")

uploaded = st.file_uploader("Upload panoramic dental X-ray (OPG)", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        st.error("Could not decode uploaded image.")
        st.stop()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with st.spinner("Running full pipeline..."):
        result = run_pipeline(image_bgr=image_bgr, device=device)

    overlay = draw_overlay(image_bgr, result.boxes, result.fdi_map)
    mask_vis = cv2.cvtColor(result.mask_binary, cv2.COLOR_GRAY2RGB)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col2:
        st.subheader("Segmentation + FDI Overlay")
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.subheader("Binary Tooth Mask")
    st.image(mask_vis, use_container_width=True)

    detected_teeth = sorted(result.classifications.keys())

    st.subheader("Clinical Output")
    st.write(f"Detected teeth count: {len(detected_teeth)}")
    st.write(f"Detected FDI teeth: {detected_teeth if detected_teeth else 'None'}")
    st.write(f"Missing teeth: {result.missing if result.missing else 'None'}")
    st.write(f"Impacted candidates: {result.impacted_candidates if result.impacted_candidates else 'None'}")
    st.write(f"Supernumerary teeth count: {result.supernumerary_count}")
    st.write(f"Structural anomaly: {result.anomaly.label} (score={result.anomaly.score})")

    st.subheader("Per-Tooth Classification")
    if result.classifications:
        st.dataframe(
            {
                "FDI Tooth": list(sorted(result.classifications.keys())),
                "Condition": [result.classifications[k] for k in sorted(result.classifications.keys())],
            },
            use_container_width=True,
        )
    else:
        st.info("No teeth could be classified.")

    report_path = generate_report_csv(
        output_dir=OUTPUTS_DIR,
        detected_teeth=detected_teeth,
        missing_teeth=result.missing,
        impacted_teeth=result.impacted_candidates,
        supernumerary_count=result.supernumerary_count,
        classifications=result.classifications,
        anomaly_label=result.anomaly.label,
        anomaly_score=result.anomaly.score,
    )

    with open(report_path, "rb") as f:
        st.download_button(
            "Download CSV Report",
            data=f,
            file_name=os.path.basename(report_path),
            mime="text/csv",
        )
