

import os
import zipfile
import random
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ─── Transforms ───────────────────────────────────────────────────────────────

def get_transforms(image_size: int = 224, split: str = "train") -> transforms.Compose:
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def get_corruption_transform(
    image_size: int = 224,
    corruption: str = "gaussian",
    severity: float = 0.1,
) -> transforms.Compose:
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    base = [
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ]

    if corruption == "gaussian":
        base.append(transforms.Lambda(
            lambda x: torch.clamp(x + torch.randn_like(x) * severity, 0, 1)
        ))
    elif corruption == "motion_blur":
        from torchvision.transforms import GaussianBlur
        k = max(3, int(severity * 30) | 1)   # kernel must be odd
        base.append(GaussianBlur(kernel_size=k, sigma=severity * 5 + 0.1))
    elif corruption == "brightness":
        base.append(transforms.Lambda(
            lambda x: torch.clamp(x + severity, 0, 1)
        ))

    base.append(transforms.Normalize(mean, std))
    return transforms.Compose(base)


# ─── Data Extraction ──────────────────────────────────────────────────────────

def extract_dataset(zip_path: str, extract_to: str) -> str:
    """Extract zip if not already done; returns the dataset root path."""
    zip_path   = Path(zip_path)
    extract_to = Path(extract_to)

    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset zip not found: {zip_path}")

    if extract_to.exists():
        print(f"[dataset] Already extracted → {extract_to}")
        return str(extract_to)

    print(f"[dataset] Extracting {zip_path} → {extract_to} …")
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)

    # If zip contains a single top-level folder, descend into it
    children = [c for c in extract_to.iterdir() if c.is_dir()]
    if len(children) == 1:
        return str(children[0])
    return str(extract_to)


# ─── Dataset Diagnostics ──────────────────────────────────────────────────────

def diagnose_dataset(root: str):
    """Print a clear summary of what ImageFolder sees at this root."""
    print("\n" + "─" * 55)
    print(f"  Dataset Diagnostic: {root}")
    print("─" * 55)
    try:
        class_dirs = sorted([
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ])
        print(f"  Class folders found : {len(class_dirs)}")
        for i, c in enumerate(class_dirs):
            n_imgs = len([
                f for f in os.listdir(os.path.join(root, c))
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            ])
            print(f"    [{i:02d}] {c:<30s}  {n_imgs} images")
    except Exception as e:
        print(f"  ERROR reading root: {e}")
    print("─" * 55 + "\n")


# ─── Dataset Splitting ────────────────────────────────────────────────────────

def split_dataset(
    root: str,
    val_fraction: float = 0.2,
    seed: int = 42,
    image_size: int = 224,
):
    """Return (train_subset, val_subset, class_names) with stratified split."""
    full    = datasets.ImageFolder(root)
    targets = np.array(full.targets)
    classes = np.unique(targets)

    rng = random.Random(seed)
    train_idx, val_idx = [], []

    for cls in classes:
        idx = np.where(targets == cls)[0].tolist()
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_fraction))
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])

    train_ds = datasets.ImageFolder(root, transform=get_transforms(image_size, "train"))
    val_ds   = datasets.ImageFolder(root, transform=get_transforms(image_size, "val"))

    return Subset(train_ds, train_idx), Subset(val_ds, val_idx), full.classes


def get_few_shot_subset(dataset: Subset, fraction: float, seed: int = 42) -> Subset:
    """Sub-sample a stratified fraction of a training Subset."""
    if fraction >= 1.0:
        return dataset

    full_dataset = dataset.dataset
    indices      = np.array(dataset.indices)
    targets      = np.array(full_dataset.targets)[indices]
    classes      = np.unique(targets)

    rng = random.Random(seed)
    selected = []
    for cls in classes:
        cls_idx = indices[targets == cls].tolist()
        rng.shuffle(cls_idx)
        n = max(1, int(len(cls_idx) * fraction))
        selected.extend(cls_idx[:n])

    return Subset(full_dataset, selected)


# ─── DataLoaders ─────────────────────────────────────────────────────────────

def make_loaders(
    root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    val_fraction: float = 0.2,
    seed: int = 42,
    image_size: int = 224,
    few_shot_fraction: float = 1.0,
):
    """Returns (train_loader, val_loader, class_names)."""
    train_ds, val_ds, class_names = split_dataset(root, val_fraction, seed, image_size)

    if few_shot_fraction < 1.0:
        train_ds = get_few_shot_subset(train_ds, few_shot_fraction, seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, class_names


def make_corruption_loader(
    root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    val_fraction: float = 0.2,
    seed: int = 42,
    image_size: int = 224,
    corruption: str = "gaussian",
    severity: float = 0.1,
):
    """Validation loader with a specific corruption applied at eval time."""
    _, val_ds_clean, class_names = split_dataset(root, val_fraction, seed, image_size)
    val_indices = val_ds_clean.indices

    corrupt_tf  = get_corruption_transform(image_size, corruption, severity)
    base_ds     = datasets.ImageFolder(root, transform=corrupt_tf)
    corrupt_ds  = Subset(base_ds, val_indices)

    loader = DataLoader(
        corrupt_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return loader, class_names