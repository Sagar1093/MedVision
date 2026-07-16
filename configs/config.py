from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = PROJECT_ROOT/"datasets"/"raw"/"archive"
MODEL_DIR = PROJECT_ROOT/"models"
CHECKPOINT_DIR = MODEL_DIR/"checkpoints"
REPORT_DIR = PROJECT_ROOT/"reports"

NUM_CLASSES = 3
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
LR_FACTOR = 0.5
LR_PATIENCE = 2

EARLY_STOPPING_PATIENCE = 5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
# resnet50, densenet121, efficientnet_b3
MODEL_NAME = "efficientnet_b3"