"""
Scenario 4.1 : Linear Probe Transfer
Scenario 4.2 : Fine-Tuning Strategies
Scenario 4.3 : Few-Shot Learning Analysis
Scenario 4.4 : Corruption Robustness Evaluation
Scenario 4.5 : Layer-Wise Feature Probing
"""

import os
import json
import numpy as np
import torch
from typing import Dict, List

import config
from dataset import make_loaders, make_corruption_loader, split_dataset, get_few_shot_subset
from models  import (
    build_model, print_efficiency_report, count_parameters,
    IntermediateFeatureExtractor,
)
from trainer import train_model, evaluate, extract_features, get_device
from visualize import (
    plot_accuracy_loss, plot_confusion_matrix, plot_embeddings,
    plot_strategy_comparison, plot_grad_norms,
    plot_few_shot, plot_corruption_robustness,
    plot_layer_probe_accuracy, plot_feature_norms,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ─── Utilities ────────────────────────────────────────────────────────────────

def _jdump(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))
    print(f"  [json] saved → {path}")


def _out(*parts) -> str:
    return os.path.join(config.RESULTS_DIR, *parts)


def _pct_unfrozen(model) -> float:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return round(100.0 * trainable / max(total, 1), 2)


# ─── 4.1 Linear Probe ─────────────────────────────────────────────────────────

def run_linear_probe(data_root: str, class_names: List[str]) -> Dict:
    print("\n" + "=" * 60)
    print("  SCENARIO 4.1 – Linear Probe Transfer")
    print("=" * 60)

    device  = get_device()
    results = {}

    train_loader, val_loader, _ = make_loaders(
        data_root,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        seed=config.SEED,
        image_size=config.IMAGE_SIZE,
    )

    for mname in config.MODELS:
        print(f"\n── {mname.upper()} ──────────────────────────")
        model = build_model(mname, config.NUM_CLASSES, pretrained=True,
                            strategy="linear_probe")
        eff   = print_efficiency_report(mname, model, config.IMAGE_SIZE)

        ckpt  = os.path.join(config.CHECKPOINTS, f"linear_probe_{mname}.pth")
        history = train_model(
            model, train_loader, val_loader,
            num_epochs=config.EPOCHS_FULL,
            lr=config.LR_LINEAR,
            weight_decay=config.WEIGHT_DECAY,
            device=device,
            model_name=f"linear_probe_{mname}",
            checkpoint_path=ckpt,
        )

        # ── plots ──────────────────────────────────────────────────────────
        plot_accuracy_loss(
            history,
            title=f"Linear Probe – {mname}",
            save_path=_out("4.1_linear_probe", mname, "acc_loss_curves.png"),
        )
        plot_confusion_matrix(
            history["best_labels"], history["best_preds"],
            class_names,
            title=f"Confusion Matrix – {mname} (linear probe)",
            save_path=_out("4.1_linear_probe", mname, "confusion_matrix.png"),
        )

        # ── feature embedding visualisation ────────────────────────────────
        model.eval()
        model.to(device)
        feats, lbls = extract_features(model, val_loader, device)
        for method in ("pca", "tsne"):
            plot_embeddings(
                feats, lbls, class_names,
                title=f"Linear Probe – {mname}",
                save_path=_out("4.1_linear_probe", mname, f"embeddings_{method}.png"),
                method=method,
            )

        results[mname] = {
            "efficiency":    eff,
            "best_val_acc":  history["best_val_acc"],
            "history":       {k: v for k, v in history.items()
                              if k not in ("best_preds", "best_labels")},
        }

    _jdump(results, _out("4.1_linear_probe", "results.json"))
    return results


# ─── 4.2 Fine-Tuning Strategies ──────────────────────────────────────────────

def run_finetuning_strategies(data_root: str, class_names: List[str]) -> Dict:
    print("\n" + "=" * 60)
    print("  SCENARIO 4.2 – Fine-Tuning Strategies")
    print("=" * 60)

    device  = get_device()
    results = {}

    train_loader, val_loader, _ = make_loaders(
        data_root,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        seed=config.SEED,
        image_size=config.IMAGE_SIZE,
    )

    strategy_lr = {
        "linear_probe":   config.LR_LINEAR,
        "last_block":     config.LR_FINETUNE,
        "full_finetune":  config.LR_FULL,
        "selective_20pct": config.LR_FINETUNE,
    }

    for mname in config.MODELS:
        results[mname] = {}
        strategy_summary = {}

        for strategy in config.FINETUNE_STRATEGIES:
            print(f"\n── {mname.upper()} | {strategy} ──────────────────")
            model = build_model(mname, config.NUM_CLASSES, pretrained=True,
                                strategy=strategy)
            pct   = _pct_unfrozen(model)
            print(f"   Unfrozen parameters: {pct:.1f}%")

            ckpt  = os.path.join(config.CHECKPOINTS, f"ft_{mname}_{strategy}.pth")
            history = train_model(
                model, train_loader, val_loader,
                num_epochs=config.EPOCHS_FULL,
                lr=strategy_lr[strategy],
                weight_decay=config.WEIGHT_DECAY,
                device=device,
                model_name=f"{mname}_{strategy}",
                collect_grad_norms=(strategy == "full_finetune"),
                checkpoint_path=ckpt,
            )

            plot_accuracy_loss(
                history,
                title=f"{mname} – {strategy}",
                save_path=_out("4.2_finetune", mname, strategy, "acc_loss.png"),
            )
            if history.get("grad_norms"):
                plot_grad_norms(
                    history["grad_norms"],
                    title=f"{mname} – {strategy}",
                    save_path=_out("4.2_finetune", mname, strategy, "grad_norms.png"),
                )

            results[mname][strategy] = {
                "pct_unfrozen":  pct,
                "best_val_acc":  history["best_val_acc"],
                "train_loss":    history["train_loss"],
                "val_loss":      history["val_loss"],
                "train_acc":     history["train_acc"],
                "val_acc":       history["val_acc"],
            }
            strategy_summary[strategy] = {
                "pct_unfrozen": pct,
                "best_val_acc": history["best_val_acc"],
            }

        plot_strategy_comparison(
            strategy_summary,
            model_name=mname,
            save_path=_out("4.2_finetune", mname, "strategy_comparison.png"),
        )

    _jdump(results, _out("4.2_finetune", "results.json"))
    return results


# ─── 4.3 Few-Shot Learning ────────────────────────────────────────────────────

def run_few_shot(data_root: str, class_names: List[str]) -> Dict:
    print("\n" + "=" * 60)
    print("  SCENARIO 4.3 – Few-Shot Learning Analysis")
    print("=" * 60)

    device  = get_device()
    results = {}
    few_shot_plot_data = {}

    # Use linear probe for fair comparison across data regimes
    for mname in config.MODELS:
        results[mname] = {}
        few_shot_plot_data[mname] = {}

        for fraction in config.FEW_SHOT_FRACTIONS:
            epochs = config.EPOCHS_FULL if fraction == 1.0 else config.EPOCHS_FEWSHOT
            print(f"\n── {mname.upper()} | {int(fraction*100)}% data | {epochs} epochs ──")

            train_loader, val_loader, _ = make_loaders(
                data_root,
                batch_size=config.BATCH_SIZE,
                num_workers=config.NUM_WORKERS,
                seed=config.SEED,
                image_size=config.IMAGE_SIZE,
                few_shot_fraction=fraction,
            )

            model = build_model(mname, config.NUM_CLASSES, pretrained=True,
                                strategy="linear_probe")
            ckpt  = os.path.join(config.CHECKPOINTS, f"fewshot_{mname}_{int(fraction*100)}pct.pth")
            history = train_model(
                model, train_loader, val_loader,
                num_epochs=epochs,
                lr=config.LR_LINEAR,
                weight_decay=config.WEIGHT_DECAY,
                device=device,
                model_name=f"{mname}_fewshot_{int(fraction*100)}pct",
                checkpoint_path=ckpt,
            )

            results[mname][fraction] = {
                "best_val_acc": history["best_val_acc"],
                "train_acc":    history["train_acc"],
                "val_acc":      history["val_acc"],
                "train_loss":   history["train_loss"],
                "val_loss":     history["val_loss"],
                "train_val_gap": float(
                    history["train_acc"][-1] - history["val_acc"][-1]
                ),
            }
            few_shot_plot_data[mname][fraction] = history["best_val_acc"]

        # Compute relative performance drop
        acc100 = results[mname][1.0]["best_val_acc"]
        acc5   = results[mname][0.05]["best_val_acc"]
        delta  = (acc100 - acc5) / max(acc100, 1e-6) * 100
        results[mname]["relative_perf_drop_pct"] = delta
        print(f"  Δ = {delta:.2f}%  (100% vs 5% data)")

    plot_few_shot(
        few_shot_plot_data,
        save_path=_out("4.3_few_shot", "few_shot_comparison.png"),
    )
    _jdump(results, _out("4.3_few_shot", "results.json"))
    return results


# ─── 4.4 Corruption Robustness ───────────────────────────────────────────────

def run_corruption_robustness(data_root: str, class_names: List[str]) -> Dict:
    print("\n" + "=" * 60)
    print("  SCENARIO 4.4 – Corruption Robustness Evaluation")
    print("=" * 60)

    device  = get_device()
    results: Dict = {}
    clean_accs: Dict = {}
    plot_corruption_data = {}
    plot_clean_data      = {}

    criterion = torch.nn.CrossEntropyLoss()

    # Build corruption spec list
    corruption_specs = []
    for sigma in config.GAUSSIAN_SIGMAS:
        corruption_specs.append(("gaussian", sigma, f"gaussian_s{sigma}"))
    corruption_specs.append(("motion_blur",  0.5,   "motion_blur"))
    corruption_specs.append(("brightness",   0.15,  "brightness_shift"))

    for mname in config.MODELS:
        ckpt = os.path.join(config.CHECKPOINTS, f"ft_{mname}_full_finetune.pth")
        # Fallback: use linear probe checkpoint
        if not os.path.exists(ckpt):
            ckpt = os.path.join(config.CHECKPOINTS, f"linear_probe_{mname}.pth")

        model = build_model(mname, config.NUM_CLASSES, pretrained=True,
                            strategy="full_finetune")
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            print(f"  [checkpoint] loaded {ckpt}")
        model = model.to(device)

        # Clean accuracy
        _, val_loader, _ = make_loaders(
            data_root,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            seed=config.SEED,
            image_size=config.IMAGE_SIZE,
        )
        clean_eval   = evaluate(model, val_loader, criterion, device)
        clean_acc    = clean_eval["val_acc"]
        clean_accs[mname]      = clean_acc
        plot_clean_data[mname] = clean_acc
        print(f"\n── {mname.upper()} | Clean Acc: {clean_acc:.2f}%")

        results[mname]           = {"clean_acc": clean_acc}
        plot_corruption_data[mname] = {}

        for corruption, severity, key in corruption_specs:
            loader, _ = make_corruption_loader(
                data_root,
                batch_size=config.BATCH_SIZE,
                num_workers=config.NUM_WORKERS,
                seed=config.SEED,
                image_size=config.IMAGE_SIZE,
                corruption=corruption,
                severity=severity,
            )
            evl  = evaluate(model, loader, criterion, device)
            acc  = evl["val_acc"]
            corr_error = 1.0 - acc / 100.0
            rel_rob    = acc / max(clean_acc, 1e-6)

            results[mname][key] = {
                "acc":                acc,
                "corruption_error":   corr_error,
                "relative_robustness": rel_rob,
            }
            plot_corruption_data[mname][key] = acc
            print(f"  {key:25s}  acc={acc:.2f}%  CE={corr_error:.4f}  RR={rel_rob:.4f}")

    plot_corruption_robustness(
        plot_corruption_data,
        plot_clean_data,
        save_path=_out("4.4_corruption", "robustness_comparison.png"),
    )
    _jdump(results, _out("4.4_corruption", "results.json"))
    return results


# ─── 4.5 Layer-Wise Feature Probing ──────────────────────────────────────────

def run_layer_probing(data_root: str, class_names: List[str]) -> Dict:
    print("\n" + "=" * 60)
    print("  SCENARIO 4.5 – Layer-Wise Feature Probing")
    print("=" * 60)

    device  = get_device()
    results = {}
    probe_acc_plot   = {}
    feature_norm_plot= {}

    # Fixed subset: 30 classes × 30 samples (for PCA plots)
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets
    from dataset import get_transforms

    full_ds   = datasets.ImageFolder(data_root, transform=get_transforms(config.IMAGE_SIZE, "val"))
    targets   = np.array(full_ds.targets)

    import random
    rng = random.Random(config.SEED)
    fixed_idx = []
    for cls in range(config.NUM_CLASSES):
        idx = np.where(targets == cls)[0].tolist()
        rng.shuffle(idx)
        fixed_idx.extend(idx[:30])

    fixed_loader = DataLoader(
        Subset(full_ds, fixed_idx),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )
    fixed_labels = targets[fixed_idx]

    # Train and val loaders for probing (fit on train, score on val)
    train_loader, val_loader, _ = make_loaders(
        data_root,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        seed=config.SEED,
        image_size=config.IMAGE_SIZE,
    )

    for mname in config.MODELS:
        print(f"\n── {mname.upper()} ──────────────────────────")
        # Load best full-finetune checkpoint if available
        ckpt = os.path.join(config.CHECKPOINTS, f"ft_{mname}_full_finetune.pth")
        if not os.path.exists(ckpt):
            ckpt = os.path.join(config.CHECKPOINTS, f"linear_probe_{mname}.pth")

        model = build_model(mname, config.NUM_CLASSES, pretrained=True,
                            strategy="full_finetune")
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model = model.to(device).eval()

        layer_specs = config.PROBE_LAYERS.get(mname, [])
        results[mname] = {}
        probe_acc_plot[mname]    = {}
        feature_norm_plot[mname] = {}

        pca_feats_per_depth = {}

        for spec in layer_specs:
            # Support both old 2-tuple (label, path) and new 3-tuple (label, tv_path, timm_path)
            if len(spec) == 3:
                depth_label, tv_path, timm_path = spec
            else:
                depth_label, tv_path = spec
                timm_path = tv_path

            # Try timm path first, fall back to torchvision path
            extractor = None
            used_path = None
            for candidate_path in [timm_path, tv_path]:
                if candidate_path == used_path:
                    continue   # don't try the same path twice
                try:
                    extractor = IntermediateFeatureExtractor(model, candidate_path)
                    used_path = candidate_path
                    print(f"  Probing layer: {candidate_path} ({depth_label})")
                    break
                except Exception as e:
                    print(f"  [warn] path '{candidate_path}' failed: {e}")
                    extractor = None

            if extractor is None:
                print(f"  [skip] No valid path found for depth '{depth_label}'")
                continue

            # Extract features: train for fitting, val for scoring, fixed for PCA
            try:
                feats_train, lbls_train = extract_features(model, train_loader, device, extractor)
                feats_val,   lbls_val   = extract_features(model, val_loader,   device, extractor)
                feats_fix,   _          = extract_features(model, fixed_loader, device, extractor)
            except Exception as e:
                print(f"  [warning] feature extraction failed: {e}")
                extractor.remove()
                continue

            extractor.remove()

            # Fit scaler on train, apply to val and fixed
            scaler = StandardScaler()
            feats_train_s = scaler.fit_transform(feats_train)
            feats_val_s   = scaler.transform(feats_val)
            feats_fix_s   = scaler.transform(feats_fix)

            # Linear probe: fit on train features, evaluate on val features
            clf = LogisticRegression(max_iter=500, C=1.0, n_jobs=-1, random_state=config.SEED)
            clf.fit(feats_train_s, lbls_train)
            probe_acc = clf.score(feats_val_s, lbls_val) * 100.0

            mean_norm  = float(np.linalg.norm(feats_val, axis=1).mean())
            probe_acc_plot[mname][depth_label]    = probe_acc
            feature_norm_plot[mname][depth_label] = mean_norm

            results[mname][depth_label] = {
                "layer":     used_path,
                "probe_acc": probe_acc,
                "mean_feat_norm": mean_norm,
            }
            pca_feats_per_depth[depth_label] = feats_fix_s
            print(f"    probe_acc={probe_acc:.2f}%  mean_norm={mean_norm:.4f}")

        # PCA 2D plots per depth
        for depth_label, feats in pca_feats_per_depth.items():
            plot_embeddings(
                feats, fixed_labels, class_names,
                title=f"{mname} – {depth_label} layer",
                save_path=_out("4.5_layer_probe", mname, f"pca_{depth_label}.png"),
                method="pca",
            )

    plot_layer_probe_accuracy(
        probe_acc_plot,
        save_path=_out("4.5_layer_probe", "probe_accuracy_vs_depth.png"),
    )
    plot_feature_norms(
        feature_norm_plot,
        save_path=_out("4.5_layer_probe", "feature_norms_vs_depth.png"),
    )
    _jdump(results, _out("4.5_layer_probe", "results.json"))
    return results