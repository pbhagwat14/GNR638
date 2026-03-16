


import os

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_ZIP      = "/home/nasir/gnr650/train_data.zip"
DATA_ROOT     = "/home/nasir/gnr650/a2/train_data"     
RESULTS_DIR   = "results"
CHECKPOINTS   = "checkpoints"

# ─── Experiment Settings ──────────────────────────────────────────────────────
SEED          = 42
NUM_CLASSES   = 30
IMAGE_SIZE    = 224
BATCH_SIZE    = 32
NUM_WORKERS   = 4

# Training epochs 
EPOCHS_FULL   = 30   # full-data training
EPOCHS_FEWSHOT= 20   # few-shot settings

# Learning rates
LR_LINEAR     = 1e-3
LR_FINETUNE   = 1e-4
LR_FULL       = 1e-4

WEIGHT_DECAY  = 1e-4
MOMENTUM      = 0.9

# ─── Model Selection ─────────────────────────────────────────────────────────
MODELS = ["efficientnet_b0", "densenet121", "resnet50"]

# ─── Few-Shot Fractions ───────────────────────────────────────────────────────
FEW_SHOT_FRACTIONS = [1.0, 0.20, 0.05]

# ─── Corruption Settings ─────────────────────────────────────────────────────
GAUSSIAN_SIGMAS = [0.05, 0.1, 0.2]



PROBE_LAYERS = {
    "resnet50": [
        ("early",  "layer1",  "layer1"),
        ("middle", "layer2",  "layer2"),
        ("deep",   "layer3",  "layer3"),
        ("final",  "layer4",  "layer4"),
    ],
    "densenet121": [
        ("early",  "features.denseblock1", "features.denseblock1"),
        ("middle", "features.denseblock2", "features.denseblock2"),
        ("deep",   "features.denseblock3", "features.denseblock3"),
        ("final",  "features.denseblock4", "features.denseblock4"),
    ],
    "efficientnet_b0": [
        # (label, torchvision_path, timm_path)
        ("early",  "features.1", "blocks.0"),
        ("middle", "features.3", "blocks.2"),
        ("deep",   "features.5", "blocks.4"),
        ("final",  "features.7", "blocks.6"),
    ],
}

# ─── Fine-Tuning Strategies ───────────────────────────────────────────────────
FINETUNE_STRATEGIES = [
    "linear_probe",
    "last_block",
    "full_finetune",
    "selective_20pct",
]