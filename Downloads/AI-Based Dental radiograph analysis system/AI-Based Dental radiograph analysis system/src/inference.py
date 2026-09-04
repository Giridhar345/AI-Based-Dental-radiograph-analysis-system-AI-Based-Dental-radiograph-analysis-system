from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from src.anomaly_detection import AnomalyResult, detect_structural_anomaly
from src.classifier import ImplantClassifier
from src.config import CLASS_NAMES, CLS_MODEL_PATH, SEG_MODEL_PATH
from src.missing_detection import (
    detect_impacted_candidates,
    detect_missing_teeth,
    detect_supernumerary_teeth,
)
from src.numbering import assign_fdi_numbers
from src.preprocess import preprocess_grayscale_image, preprocess_rgb_for_classifier
from src.segmentation import UNet
from src.tooth_detection import detect_teeth_boxes, map_centers_to_boxes, postprocess_mask, tooth_centers


Center = Tuple[int, int]
Box = Tuple[int, int, int, int]


@dataclass
class PipelineOutput:
    mask_binary: np.ndarray
    boxes: List[Box]
    fdi_map: Dict[Center, int]
    missing: List[int]
    impacted_candidates: List[int]
    supernumerary_count: int
    classifications: Dict[int, str]
    anomaly: AnomalyResult


def load_segmentation_model(device: torch.device) -> UNet:
    model = UNet(in_channels=1, out_channels=1).to(device)
    if Path(SEG_MODEL_PATH).exists():
        state = torch.load(SEG_MODEL_PATH, map_location=device)
        model.load_state_dict(state)
    model.eval()
    return model


def load_classifier_model(device: torch.device) -> ImplantClassifier:
    model = ImplantClassifier(num_classes=len(CLASS_NAMES)).to(device)
    if Path(CLS_MODEL_PATH).exists():
        state = torch.load(CLS_MODEL_PATH, map_location=device)
        model.load_state_dict(state)
    model.eval()
    return model


def segment_teeth(image_bgr: np.ndarray, model: UNet, device: torch.device, img_size: int = 512) -> np.ndarray:
    tensor = preprocess_grayscale_image(image_bgr, img_size=img_size).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    return postprocess_mask(probs, threshold=0.5)


def classify_tooth_crop(crop: np.ndarray, classifier: ImplantClassifier, device: torch.device) -> str:
    if crop.size == 0:
        return "Normal"
    tensor = preprocess_rgb_for_classifier(crop, img_size=224).to(device)
    with torch.no_grad():
        logits = classifier(tensor)
        pred = int(torch.argmax(logits, dim=1).item())
    return CLASS_NAMES[pred]


def run_pipeline(image_bgr: np.ndarray, device: torch.device) -> PipelineOutput:
    seg_model = load_segmentation_model(device)
    cls_model = load_classifier_model(device)

    mask_binary = segment_teeth(image_bgr, seg_model, device=device, img_size=512)
    boxes = detect_teeth_boxes(mask_binary, min_area=150)
    centers = tooth_centers(boxes)
    fdi_map = assign_fdi_numbers(centers)
    center_to_box = map_centers_to_boxes(centers, boxes)

    missing = detect_missing_teeth(fdi_map)
    impacted_candidates = detect_impacted_candidates(fdi_map)
    supernumerary_count = detect_supernumerary_teeth(fdi_map)

    classifications: Dict[int, str] = {}
    for center, fdi in fdi_map.items():
        x, y, w, h = center_to_box[center]
        x0, y0 = max(0, x - 4), max(0, y - 4)
        x1, y1 = min(image_bgr.shape[1], x + w + 4), min(image_bgr.shape[0], y + h + 4)
        crop = image_bgr[y0:y1, x0:x1]
        classifications[fdi] = classify_tooth_crop(crop, cls_model, device=device)

    anomaly = detect_structural_anomaly(image_bgr)

    return PipelineOutput(
        mask_binary=mask_binary,
        boxes=boxes,
        fdi_map=fdi_map,
        missing=missing,
        impacted_candidates=impacted_candidates,
        supernumerary_count=supernumerary_count,
        classifications=classifications,
        anomaly=anomaly,
    )
