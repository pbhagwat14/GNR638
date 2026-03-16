

import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple


# ─── Device ──────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─── Metric Helpers ──────────────────────────────────────────────────────────

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val, n=1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    pred = output.argmax(dim=1)
    return (pred == target).float().mean().item() * 100.0


# ─── Gradient Norm ───────────────────────────────────────────────────────────

def grad_norm_per_layer(model: nn.Module) -> Dict[str, float]:
    norms = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            norms[name] = param.grad.norm(2).item()
    return norms


# ─── Single Epoch Routines ────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    model_name: str = "",
    collect_grad_norms: bool = False,
) -> Dict:
    model.train()
    loss_m  = AverageMeter()
    acc_m   = AverageMeter()
    grad_norms_accum: Dict[str, List[float]] = {}

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # Inception v3 returns (logits, aux_logits) during training
        outputs = model(images)
        if isinstance(outputs, tuple):
            logits, aux = outputs[0], outputs[1]
            loss = criterion(logits, labels) + 0.4 * criterion(aux, labels)
        else:
            logits = outputs
            loss   = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()

        if collect_grad_norms:
            for k, v in grad_norm_per_layer(model).items():
                grad_norms_accum.setdefault(k, []).append(v)

        optimizer.step()

        acc = accuracy(logits, labels)
        loss_m.update(loss.item(), images.size(0))
        acc_m.update(acc, images.size(0))

    result = {"train_loss": loss_m.avg, "train_acc": acc_m.avg}
    if collect_grad_norms:
        result["grad_norms"] = {k: float(np.mean(v)) for k, v in grad_norms_accum.items()}
    return result


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict:
    model.eval()
    loss_m = AverageMeter()
    acc_m  = AverageMeter()
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        loss = criterion(outputs, labels)
        acc  = accuracy(outputs, labels)
        loss_m.update(loss.item(), images.size(0))
        acc_m.update(acc, images.size(0))

        all_preds.append(outputs.argmax(1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    return {
        "val_loss": loss_m.avg,
        "val_acc":  acc_m.avg,
        "preds":    all_preds.tolist(),
        "labels":   all_labels.tolist(),
    }


# ─── Full Training Loop ───────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    model_name: str = "",
    collect_grad_norms: bool = False,
    checkpoint_path: Optional[str] = None,
) -> Dict:
    """Full training loop. Returns history dict with per-epoch metrics."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Only pass parameters that require grad
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError(f"[train_model] No trainable parameters for '{model_name}'!")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history: Dict[str, List] = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "grad_norms": [],
    }
    best_acc    = 0.0
    best_preds  = None
    best_labels = None

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            model_name=model_name,
            collect_grad_norms=collect_grad_norms,
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_metrics["train_loss"])
        history["train_acc"].append(train_metrics["train_acc"])
        history["val_loss"].append(val_metrics["val_loss"])
        history["val_acc"].append(val_metrics["val_acc"])
        if collect_grad_norms and "grad_norms" in train_metrics:
            history["grad_norms"].append(train_metrics["grad_norms"])

        elapsed = time.time() - t0
        print(
            f"[{model_name}] Epoch {epoch:3d}/{num_epochs} | "
            f"Train Loss {train_metrics['train_loss']:.4f}  Acc {train_metrics['train_acc']:.2f}% | "
            f"Val Loss {val_metrics['val_loss']:.4f}  Acc {val_metrics['val_acc']:.2f}% | "
            f"Time {elapsed:.1f}s"
        )

        if val_metrics["val_acc"] > best_acc:
            best_acc    = val_metrics["val_acc"]
            best_preds  = val_metrics["preds"]
            best_labels = val_metrics["labels"]
            if checkpoint_path:
                import os
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)

    history["best_val_acc"]  = best_acc
    history["best_preds"]    = best_preds
    history["best_labels"]   = best_labels
    return history


# ─── Feature Extraction ───────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(
    model: nn.Module,
    loader,
    device: torch.device,
    layer_extractor=None,
) -> Tuple[np.ndarray, np.ndarray]:
   
    model.eval()

    if layer_extractor is not None:
        # Use the provided hook-based extractor
        feats_list, labels_list = [], []
        for images, labels in loader:
            images = images.to(device)
            out = layer_extractor(images)
            if isinstance(out, tuple):
                out = out[0]
            if out.dim() > 2:
                out = out.mean(dim=list(range(2, out.dim())))
            feats_list.append(out.cpu().numpy())
            labels_list.append(labels.numpy())
        return np.concatenate(feats_list), np.concatenate(labels_list)

   
    feats_list, labels_list = [], []
    hook_output = [None]
    hook_handle  = None

    # Find the last non-Linear child to hook (the pooling / flatten stage)
    penultimate = _find_penultimate_module(model)
    if penultimate is not None:
        def _hook(m, i, o):
            x = o
            if isinstance(x, tuple):
                x = x[0]
            if x.dim() == 4:
                x = x.mean(dim=[2, 3])
            elif x.dim() == 3:
                x = x.mean(dim=1)
            hook_output[0] = x.detach()

        hook_handle = penultimate.register_forward_hook(_hook)

    try:
        for images, labels in loader:
            images = images.to(device)
            model(images)
            out = hook_output[0]
            if out is None:
                raise RuntimeError("Penultimate hook not triggered.")
            feats_list.append(out.cpu().numpy())
            labels_list.append(labels.numpy())
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    return np.concatenate(feats_list), np.concatenate(labels_list)


def _find_penultimate_module(model: nn.Module) -> Optional[nn.Module]:
 
   
    children = list(model.named_children())
    # Skip known head names
    head_names = {"fc", "classifier", "head", "AuxLogits"}

    for name, module in reversed(children):
        if name in head_names:
            continue
        # Return first non-head module we hit (deepest backbone component)
        return module

    
    return model