# src/config.py

import torch

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data
IMAGE_SIZE = 160
BATCH_SIZE = 16
NUM_WORKERS = 2

# Training
EPOCHS = 2
LEARNING_RATE = 1e-3

# Dataset
NUM_CLASSES = 7
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

RANDOM_SEED = 123

SAMPLE_FRAC = 0.30    # use 30% dataset
EPOCHS = 10
LEARNING_RATE = 1e-4

