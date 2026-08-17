import os
import torch
from pathlib import Path

# ----------------------------
# Project Root & Paths
# ----------------------------

# ROOT resolves to the project root (parent of the src directory).
ROOT = Path(__file__).resolve().parents[1]

TRAIN_ROOT = ROOT / "dataset" / "train" / "train" / "train"
TRAIN_NOISY = TRAIN_ROOT / "NoisyLR"
TRAIN_GT = TRAIN_ROOT / "GT"

TEST_DIR = ROOT / "dataset" / "test" / "Test_NoisyLR" / "NoisyLR"

CHECKPOINT_DIR = ROOT / "checkpoints"
OUTPUT_DIR = ROOT / "outputs"
LOG_DIR = ROOT / "logs"

# ----------------------------
# Training Hyper-parameters
# ----------------------------

BATCH_SIZE = 8
NUM_WORKERS = 0 if os.name == "nt" else 4
PIN_MEMORY = False if os.name == "nt" else True

EPOCHS = 100

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5

# Portion of the training set held out for validation.
VALID_SPLIT = 0.05

# How often (in epochs) to save a checkpoint.
SAVE_EVERY = 5

# How many batches between progress-log updates.
PRINT_EVERY = 20

# ----------------------------
# Optimizer / Scheduler
# ----------------------------

OPTIMIZER = "adamw"

# ----------------------------
# Mixed Precision / Stability
# ----------------------------

# Use Automatic Mixed Precision (AMP) on CUDA. Ignored on CPU.
USE_AMP = True

# Max norm for gradient clipping (0 disables clipping).
GRAD_CLIP_MAX_NORM = 1.0

# ----------------------------
# Early Stopping
# ----------------------------

EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 15

# ----------------------------
# Device
# ----------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Reproducibility
# ----------------------------

SEED = 42

# ----------------------------
# Model
# ----------------------------

IN_CHANNELS = 1
OUT_CHANNELS = 1
UPSCALE = 2

# ----------------------------
# Logging
# ----------------------------

LOG_LEVEL = "INFO"

# ----------------------------
# Inference
# ----------------------------

# Use Test-Time Augmentation (flips) for potentially better results.
USE_TTA = False
