from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
MASKS_DIR = DATA_DIR / "masks"
CLASSIFIER_DIR = DATA_DIR / "classifier"

MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

SEG_MODEL_PATH = MODELS_DIR / "segmentation_model.pth"
CLS_MODEL_PATH = MODELS_DIR / "classifier_model.pth"

CLASS_NAMES = ["Normal", "Cavity-Involved", "Impacted", "Filling"]

FDI_EXPECTED = [
    11, 12, 13, 14, 15, 16, 17, 18,
    21, 22, 23, 24, 25, 26, 27, 28,
    31, 32, 33, 34, 35, 36, 37, 38,
    41, 42, 43, 44, 45, 46, 47, 48,
]
