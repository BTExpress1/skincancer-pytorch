# src/model.py
import torch.nn as nn
from torchvision import models
from src.config import NUM_CLASSES

def build_model(unfreeze_backbone=False):
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)

    # Freeze backbone by default
    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, NUM_CLASSES)
    )

    if unfreeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = True

    return model