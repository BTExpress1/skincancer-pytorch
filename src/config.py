# src/config.py

# Device
DEVICE = "cpu"   # local proof first

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

# Debug / local control
SAMPLE_FRAC = 0.10   # use 10% of data for quick run