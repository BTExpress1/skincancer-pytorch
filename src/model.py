# src/model.py

import torch.nn as nn
from torchvision import models

from src.config import NUM_CLASSES


def build_model():
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

    in_features = model.classifier[-1].in_features

    model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)

    return model