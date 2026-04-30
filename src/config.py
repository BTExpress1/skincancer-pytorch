# src/config.py
import torch

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data
IMAGE_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 12

# Classes
NUM_CLASSES = 7
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# Training
EPOCHS = 20
UNFREEZE_EPOCH = 7
LEARNING_RATE = 1e-4
RANDOM_SEED = 123
SAMPLE_FRAC = 1.0
