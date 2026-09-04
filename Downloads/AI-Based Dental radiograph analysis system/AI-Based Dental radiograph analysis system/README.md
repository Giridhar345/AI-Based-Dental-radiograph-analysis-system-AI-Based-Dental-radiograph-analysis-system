# Use Case-2: AI-Based Dental Radiograph Analysis System

End-to-end MVP for panoramic dental X-ray (OPG) analysis with:

- Tooth segmentation (U-Net)
- Tooth detection from mask
- FDI numbering (permanent dentition)
- Missing/impacted/supernumerary rule-based checks
- Tooth condition classification (`Normal`, `Cavity-Involved`, `Impacted`, `Filling`)
- Structural anomaly score (`peri-implantitis` / `bone loss` proxy)
- Streamlit dashboard + downloadable report

## 1) Project structure

```
.
├── dashboard/
│   └── app.py
├── data/
│   ├── images/
│   ├── masks/
│   └── classifier/
│       ├── train/
│       └── val/
├── models/
├── outputs/
├── src/
│   ├── __init__.py
│   ├── anomaly_detection.py
│   ├── classifier.py
│   ├── config.py
│   ├── dataset.py
│   ├── inference.py
│   ├── missing_detection.py
│   ├── numbering.py
│   ├── preprocess.py
│   ├── segmentation.py
│   └── tooth_detection.py
├── utils/
│   ├── __init__.py
│   └── report_generator.py
├── train_classifier.py
├── train_segmentation.py
└── requirements.txt
```

## 2) Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 3) Data expectations

### Segmentation
- Input images: `data/images/*.png|jpg|jpeg`
- Corresponding masks: `data/masks/<same_filename>.png`
- Binary masks (tooth=1, background=0) or grayscale (thresholded internally)

### Classification
- ImageFolder format:

```
data/classifier/train/
  Normal/
  Cavity-Involved/
  Impacted/
  Filling/
data/classifier/val/
  Normal/
  Cavity-Involved/
  Impacted/
  Filling/
```

## 4) Train models

```bash
python train_segmentation.py --epochs 30 --batch-size 4 --img-size 512
python train_classifier.py --epochs 20 --batch-size 16 --img-size 224
```

Saved models:
- `models/segmentation_model.pth`
- `models/classifier_model.pth`

## 5) Run dashboard

```bash
streamlit run dashboard/app.py
```

## 6) Notes

- Current pipeline is a strong MVP baseline.
- Clinical use requires rigorous validation, calibrated thresholds, and institutional review.
- FDI numbering uses permanent teeth expected set:
  `11-18, 21-28, 31-38, 41-48`.
