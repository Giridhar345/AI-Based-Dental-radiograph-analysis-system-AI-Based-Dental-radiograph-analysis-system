from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.config import IMAGES_DIR, MASKS_DIR, MODELS_DIR, SEG_MODEL_PATH
from src.dataset import SegmentationDataset
from src.segmentation import UNet


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.view(-1)
    targets = targets.view(-1)
    intersection = (probs * targets).sum()
    dice = (2.0 * intersection + smooth) / (probs.sum() + targets.sum() + smooth)
    return 1.0 - dice


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    dataset = SegmentationDataset(Path(IMAGES_DIR), Path(MASKS_DIR), img_size=args.img_size)
    if len(dataset) == 0:
        raise RuntimeError("No segmentation data found in data/images and data/masks")

    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=1).to(device)

    bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"[Seg] Epoch {epoch}/{args.epochs}"):
            images, masks = images.to(device), masks.to(device)

            logits = model(images)
            loss = 0.5 * bce(logits, masks) + 0.5 * dice_loss(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                loss = 0.5 * bce(logits, masks) + 0.5 * dice_loss(logits, masks)
                val_loss += loss.item()

        val_loss /= max(1, len(val_loader))
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), SEG_MODEL_PATH)
            print(f"Saved best segmentation model: {SEG_MODEL_PATH}")


if __name__ == "__main__":
    main()
