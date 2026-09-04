from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class ImplantClassifier(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
