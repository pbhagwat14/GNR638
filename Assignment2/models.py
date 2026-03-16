

import copy
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional


# ─── Model Factory ────────────────────────────────────────────────────────────

def build_model(
    name: str,
    num_classes: int = 30,
    pretrained: bool = True,
    strategy: str = "linear_probe",
) -> nn.Module:
    """
    Load a pretrained backbone, replace the head, apply freezing strategy.
    Returns the model ready for training.
    """
    model, head_attr = _load_and_replace_head(name, num_classes, pretrained)

    # Sanity check: the head must have at least one parameter before freezing
    _verify_head(model, head_attr)

    model = _apply_strategy(model, name, strategy, head_attr)

    # Final sanity: at least one parameter must require grad
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable == 0:
        raise RuntimeError(
            f"[build_model] No trainable parameters after strategy '{strategy}' "
            f"on {name}. head_attr='{head_attr}'."
        )
    print(f"  [build_model] {name} | strategy={strategy} | "
          f"trainable={trainable:,} / "
          f"{sum(p.numel() for p in model.parameters()):,} params")
    return model


# ─── Backend Loaders ─────────────────────────────────────────────────────────

def _load_and_replace_head(
    name: str, num_classes: int, pretrained: bool
) -> Tuple[nn.Module, str]:
    """Returns (model, head_attribute_name)."""
    try:
        return _load_timm(name, num_classes, pretrained)
    except Exception as e:
        print(f"[model] timm failed ({e}), falling back to torchvision …")
        return _load_torchvision(name, num_classes, pretrained)


# ── timm path ────────────────────────────────────────────────────────────────

_TIMM_NAMES = {
    "resnet50":        "resnet50",
    "efficientnet_b0": "efficientnet_b0",
    "densenet121":     "densenet121",
    "inception_v3":    "inception_v3",
    "convnext_tiny":   "convnext_tiny",
}


def _load_timm(name: str, num_classes: int, pretrained: bool) -> Tuple[nn.Module, str]:
    import timm

    tname = _TIMM_NAMES.get(name, name)

    # Load WITH num_classes so timm builds the correct head immediately
    model = timm.create_model(
        tname,
        pretrained=pretrained,
        num_classes=num_classes,
    )

    head_attr = _detect_head_attr(model, name)
    print(f"  [timm] loaded '{tname}' | head_attr='{head_attr}'")
    return model, head_attr


def _detect_head_attr(model: nn.Module, name: str) -> str:
    """
    Robustly detect the attribute name of the classification head.
    Checks common names first, then falls back to scanning children.
    """
    for attr in ("head", "fc", "classifier", "head_drop"):
        module = getattr(model, attr, None)
        if module is not None and _has_linear(module):
            return attr

    # Fallback: last child with a Linear layer
    for attr, module in reversed(list(model.named_children())):
        if _has_linear(module):
            return attr

    raise RuntimeError(
        f"Cannot detect head attribute for '{name}'. "
        f"Children: {[n for n, _ in model.named_children()]}"
    )


def _has_linear(m: nn.Module) -> bool:
    return any(isinstance(c, nn.Linear) for c in m.modules())


# ── torchvision path ─────────────────────────────────────────────────────────

def _load_torchvision(name: str, num_classes: int, pretrained: bool) -> Tuple[nn.Module, str]:
    from torchvision import models

    weights_map = {
        "resnet50":        models.ResNet50_Weights.IMAGENET1K_V1,
        "efficientnet_b0": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        "densenet121":     models.DenseNet121_Weights.IMAGENET1K_V1,
        "inception_v3":    models.Inception_V3_Weights.IMAGENET1K_V1,
        "convnext_tiny":   models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
    }
    loaders = {
        "resnet50":        models.resnet50,
        "efficientnet_b0": models.efficientnet_b0,
        "densenet121":     models.densenet121,
        "inception_v3":    models.inception_v3,
        "convnext_tiny":   models.convnext_tiny,
    }
    if name not in loaders:
        raise ValueError(f"Unknown model: {name}")

    w     = weights_map[name] if pretrained else None
    model = loaders[name](weights=w)
    head_attr = _replace_torchvision_head(model, name, num_classes)
    print(f"  [torchvision] loaded '{name}' | head_attr='{head_attr}'")
    return model, head_attr


def _replace_torchvision_head(model: nn.Module, name: str, num_classes: int) -> str:
    if name == "resnet50":
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
        return "fc"
    elif name == "efficientnet_b0":
        in_f = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_f, num_classes),
        )
        return "classifier"
    elif name == "densenet121":
        in_f = model.classifier.in_features
        model.classifier = nn.Linear(in_f, num_classes)
        return "classifier"
    elif name == "inception_v3":
        in_f = model.fc.in_features
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
        model.fc = nn.Linear(in_f, num_classes)
        return "fc"
    elif name == "convnext_tiny":
        in_f = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_f, num_classes)
        return "classifier"
    raise ValueError(f"Unknown torchvision model: {name}")


# ─── Freezing Strategies ─────────────────────────────────────────────────────

def _apply_strategy(
    model: nn.Module, name: str, strategy: str, head_attr: str
) -> nn.Module:
    if strategy == "linear_probe":
        _freeze_all(model)
        _unfreeze_module(model, head_attr)

    elif strategy == "last_block":
        _freeze_all(model)
        _unfreeze_last_block(model, name)
        _unfreeze_module(model, head_attr)

    elif strategy == "full_finetune":
        _unfreeze_all(model)

    elif strategy == "selective_20pct":
        _freeze_all(model)
        _unfreeze_selective_20pct(model, head_attr)
        _unfreeze_module(model, head_attr)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return model


def _freeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False


def _unfreeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = True


def _unfreeze_module(model: nn.Module, attr: str):
    """Unfreeze all parameters reachable via dotted attribute path."""
    parts  = attr.split(".")
    module = model
    for part in parts:
        module = getattr(module, part, None)
        if module is None:
            available = [n for n, _ in model.named_children()]
            raise RuntimeError(
                f"[_unfreeze_module] '{attr}' not found. "
                f"Top-level children: {available}"
            )
    count = 0
    for p in module.parameters():
        p.requires_grad = True
        count += p.numel()
    if count == 0:
        raise RuntimeError(
            f"[_unfreeze_module] module at '{attr}' has 0 parameters!"
        )


def _unfreeze_last_block(model: nn.Module, name: str):
    """Unfreeze only the last major feature block of the backbone."""
    # Preferred dotted paths per architecture
    last_block_paths = {
        "resnet50":        "layer4",
        "efficientnet_b0": "blocks",
        "densenet121":     "features.denseblock4",
        "inception_v3":    "Mixed_7c",
        "convnext_tiny":   "stages",
    }
    path = last_block_paths.get(name)
    if path is None:
        print(f"  [warn] No last_block mapping for '{name}'; skipping.")
        return

    parts  = path.split(".")
    module = model
    for part in parts:
        module = getattr(module, part, None)
        if module is None:
            print(f"  [warn] last_block path '{path}' not found on model; skipping.")
            return

    # For sequential containers, only unfreeze the LAST sub-block
    if isinstance(module, (nn.Sequential, nn.ModuleList)):
        children = list(module.children())
        if children:
            for p in children[-1].parameters():
                p.requires_grad = True
            return

    for p in module.parameters():
        p.requires_grad = True


def _unfreeze_selective_20pct(model: nn.Module, head_attr: str):
    """
    Unfreeze the deepest ~20% of backbone parameters.

    Rationale (Yosinski et al., 2014):
      Deeper layers encode task-specific, high-level semantics and yield the
      highest accuracy gain per additional trainable parameter. Unfreezing from
      the deepest layer upward satisfies the assignment's efficiency bonus.
    """
    head_top = head_attr.split(".")[0]
    aux_names = {head_top, "AuxLogits"}

    backbone_children = [
        (n, m) for n, m in model.named_children()
        if n not in aux_names
    ]

    total_backbone = sum(
        p.numel() for _, m in backbone_children for p in m.parameters()
    )
    budget = int(0.20 * total_backbone)

    unfrozen = 0
    for _, bmodule in reversed(backbone_children):
        if unfrozen >= budget:
            break
        # Walk named parameters in reverse (deepest first)
        params = list(bmodule.named_parameters())
        for _, param in reversed(params):
            if unfrozen >= budget:
                break
            param.requires_grad = True
            unfrozen += param.numel()

    pct = 100.0 * unfrozen / max(total_backbone, 1)
    print(f"  [selective] unfrozen {unfrozen:,} / {total_backbone:,} backbone params ({pct:.1f}%)")


# ─── Verification ─────────────────────────────────────────────────────────────

def _verify_head(model: nn.Module, head_attr: str):
    parts  = head_attr.split(".")
    module = model
    for part in parts:
        module = getattr(module, part, None)
        if module is None:
            available = [n for n, _ in model.named_children()]
            raise RuntimeError(
                f"Head attribute '{head_attr}' not found. "
                f"Top-level children: {available}"
            )
    n_params = sum(p.numel() for p in module.parameters())
    if n_params == 0:
        raise RuntimeError(
            f"Head module at '{head_attr}' has 0 parameters — model construction failed."
        )


# ─── Efficiency Metrics ───────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> Dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable}


def compute_macs_flops(model: nn.Module, image_size: int = 224) -> Dict:
    dummy = torch.randn(1, 3, image_size, image_size)
    m = copy.deepcopy(model).cpu().eval()

    try:
        from thop import profile
        macs, _ = profile(m, inputs=(dummy,), verbose=False)
        return {"MACs": float(macs), "FLOPs": float(2 * macs), "source": "thop"}
    except Exception:
        pass

    try:
        from ptflops import get_model_complexity_info
        macs, _ = get_model_complexity_info(
            m, (3, image_size, image_size),
            as_strings=False, print_per_layer_stat=False,
        )
        return {"MACs": float(macs), "FLOPs": float(2 * macs), "source": "ptflops"}
    except Exception:
        pass

    return {"MACs": -1.0, "FLOPs": -1.0, "source": "unavailable"}


def print_efficiency_report(model_name: str, model: nn.Module, image_size: int = 224) -> Dict:
    params = count_parameters(model)
    macs   = compute_macs_flops(model, image_size)
    sep    = "─" * 58
    print(f"\n{sep}")
    print(f"  Efficiency Report: {model_name.upper()}")
    print(sep)
    print(f"  Total parameters   : {params['total_params']:>15,}")
    print(f"  Trainable params   : {params['trainable_params']:>15,}")
    if macs.get("MACs", -1) >= 0:
        print(f"  MACs               : {macs['MACs']:>15,.0f}  [{macs['source']}]")
        print(f"  FLOPs              : {macs['FLOPs']:>15,.0f}")
    print(sep)
    return {**params, **macs}


# ─── Feature Extractor Hook ───────────────────────────────────────────────────

class IntermediateFeatureExtractor(nn.Module):
    """
    Attaches a forward hook to a named sub-module and returns its output
    as a flat (B, D) tensor (after global average pooling if spatial).

    Usage:
        ext = IntermediateFeatureExtractor(model, "layer3")
        feats = ext(x)
        ext.remove()    # remove hook when done
    """

    def __init__(self, model: nn.Module, layer_path: str):
        super().__init__()
        self.model      = model
        self.layer_path = layer_path
        self._features  = None
        self._hook_handle = None
        self._register_hook()

    def _register_hook(self):
        parts  = self.layer_path.split(".")
        module = self.model
        for part in parts:
            module = getattr(module, part, None)
            if module is None:
                raise RuntimeError(
                    f"Layer path '{self.layer_path}' not found on model."
                )
        self._hook_handle = module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, inp, output):
        if isinstance(output, tuple):
            output = output[0]
        self._features = output.detach()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._features = None
        self.model(x)
        if self._features is None:
            raise RuntimeError(f"Hook for '{self.layer_path}' was not triggered.")
        feat = self._features
        if feat.dim() == 4:
            feat = feat.mean(dim=[2, 3])
        elif feat.dim() == 3:
            feat = feat.mean(dim=1)
        return feat

    def remove(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None