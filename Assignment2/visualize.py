

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import confusion_matrix
import seaborn as sns


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _save(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [plot] saved → {path}")


# ─── Accuracy / Loss Curves ───────────────────────────────────────────────────

def plot_accuracy_loss(
    history: dict,
    title: str,
    save_path: str,
):
    epochs = range(1, len(history["train_acc"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_acc"], label="Train")
    axes[0].plot(epochs, history["val_acc"],   label="Val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title(f"{title} – Accuracy"); axes[0].legend(); axes[0].grid(True)

    axes[1].plot(epochs, history["train_loss"], label="Train")
    axes[1].plot(epochs, history["val_loss"],   label="Val")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].set_title(f"{title} – Loss"); axes[1].legend(); axes[1].grid(True)

    fig.tight_layout()
    _save(fig, save_path)


# ─── Confusion Matrix ─────────────────────────────────────────────────────────

def plot_confusion_matrix(
    labels: list,
    preds: list,
    class_names: list,
    title: str,
    save_path: str,
):
    cm = confusion_matrix(labels, preds)
    n  = len(class_names)

    # Scale figure so every cell is readable
    cell_size = 0.6
    fig_w = max(14, n * cell_size + 3)
    fig_h = max(12, n * cell_size + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Draw heatmap (no seaborn annotations — we draw our own)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    # Tick labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, rotation=0, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=13, pad=14)

    # Annotate every non-zero cell with its count
    thresh    = cm.max() / 2.0                       # white text above this
    font_size = max(5, min(10, int(180 / n)))        # shrink font for big matrices

    for i in range(n):
        for j in range(n):
            value = int(cm[i, j])
            if value == 0:
                continue                             # skip zeros for clarity
            color = "white" if value > thresh else "black"
            ax.text(
                j, i, str(value),
                ha="center", va="center",
                fontsize=font_size,
                fontweight="bold" if i == j else "normal",
                color=color,
            )

    # Subtle grid lines between cells
    ax.set_xticks([x - 0.5 for x in range(1, n)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, n)], minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", length=0)

    fig.tight_layout()
    _save(fig, save_path)


# ─── 2D Embedding Plots (PCA / t-SNE / UMAP) ─────────────────────────────────

def _reduce(features: np.ndarray, method: str = "pca", n_components: int = 2) -> np.ndarray:
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=n_components, random_state=42).fit_transform(features)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        perp = min(30, max(5, len(features) // 5))
        return TSNE(n_components=n_components, perplexity=perp, random_state=42,
                    n_iter=500).fit_transform(features)
    elif method == "umap":
        import umap
        return umap.UMAP(n_components=n_components, random_state=42).fit_transform(features)
    raise ValueError(f"Unknown reduction method: {method}")


def plot_embeddings(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: list,
    title: str,
    save_path: str,
    method: str = "pca",
):
    reduced = _reduce(features, method)
    n_cls   = len(class_names)
    cmap    = plt.get_cmap("tab20", n_cls)

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, cname in enumerate(class_names):
        mask = labels == i
        if mask.sum() == 0:
            continue
        ax.scatter(
            reduced[mask, 0], reduced[mask, 1],
            c=[cmap(i)], label=cname, s=12, alpha=0.7,
        )
    ax.set_title(f"{title} ({method.upper()})")
    ax.set_xlabel("Component 1"); ax.set_ylabel("Component 2")
    if n_cls <= 20:
        ax.legend(fontsize=6, ncol=2, loc="best")
    _save(fig, save_path)


# ─── Fine-Tuning Strategy Comparison ─────────────────────────────────────────

def plot_strategy_comparison(
    results: dict,   # {strategy: {"pct_unfrozen": float, "best_val_acc": float}}
    model_name: str,
    save_path: str,
):
    strategies = list(results.keys())
    x_pct  = [results[s].get("pct_unfrozen", 0) for s in strategies]
    y_acc  = [results[s].get("best_val_acc",  0) for s in strategies]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x_pct, y_acc, s=80, zorder=3)
    for i, s in enumerate(strategies):
        ax.annotate(s, (x_pct[i], y_acc[i]), textcoords="offset points",
                    xytext=(6, 3), fontsize=8)
    ax.plot(x_pct, y_acc, "k--", alpha=0.4)
    ax.set_xlabel("% Unfrozen Parameters")
    ax.set_ylabel("Best Val Accuracy (%)")
    ax.set_title(f"{model_name} – Accuracy vs % Unfrozen Params")
    ax.grid(True)
    _save(fig, save_path)


# ─── Gradient Norm Heatmap ────────────────────────────────────────────────────

def plot_grad_norms(
    grad_norms: list,    # list of dicts per epoch
    title: str,
    save_path: str,
    max_layers: int = 30,
):
    if not grad_norms:
        return
    # Collect all layer names
    all_keys = list(grad_norms[0].keys())[:max_layers]
    matrix   = np.array([[ep.get(k, 0) for k in all_keys] for ep in grad_norms])
    matrix   = np.log1p(matrix)   # log-scale for readability

    fig, ax = plt.subplots(figsize=(max(10, len(all_keys) * 0.3), 5))
    im = ax.imshow(matrix.T, aspect="auto", cmap="viridis")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Layer")
    ax.set_yticks(range(len(all_keys)))
    ax.set_yticklabels(all_keys, fontsize=5)
    ax.set_title(f"{title} – Gradient Norms (log scale)")
    fig.colorbar(im, ax=ax, label="log(1+norm)")
    _save(fig, save_path)


# ─── Few-Shot Accuracy Bar Chart ─────────────────────────────────────────────

def plot_few_shot(
    results: dict,   # {model_name: {fraction: acc}}
    save_path: str,
):
    fractions = ["5%", "20%", "100%"]
    frac_keys = [0.05, 0.20, 1.00]
    model_names = list(results.keys())
    x = np.arange(len(fractions))
    width = 0.8 / max(len(model_names), 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(model_names):
        accs = [results[m].get(fk, 0) for fk in frac_keys]
        ax.bar(x + i * width, accs, width, label=m)

    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(fractions)
    ax.set_xlabel("Training Data Fraction")
    ax.set_ylabel("Val Accuracy (%)")
    ax.set_title("Few-Shot Learning Comparison")
    ax.legend(); ax.grid(axis="y", alpha=0.5)
    _save(fig, save_path)


# ─── Corruption Robustness ────────────────────────────────────────────────────

def plot_corruption_robustness(
    results: dict,   # {model: {corruption_key: acc}}
    clean_results: dict,  # {model: clean_acc}
    save_path: str,
):
    model_names = list(results.keys())
    all_corruptions = list(next(iter(results.values())).keys())

    fig, ax = plt.subplots(figsize=(max(10, len(all_corruptions)), 5))
    x      = np.arange(len(all_corruptions))
    width  = 0.8 / max(len(model_names), 1)

    for i, m in enumerate(model_names):
        accs = [results[m].get(ck, 0) for ck in all_corruptions]
        ax.bar(x + i * width, accs, width, label=m)

    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(all_corruptions, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Val Accuracy (%)")
    ax.set_title("Corruption Robustness Comparison")
    ax.legend(); ax.grid(axis="y", alpha=0.5)
    _save(fig, save_path)


# ─── Layer-Wise Probing ───────────────────────────────────────────────────────

def plot_layer_probe_accuracy(
    results: dict,   # {model: {depth_label: acc}}
    save_path: str,
):
    fig, ax = plt.subplots(figsize=(9, 5))
    for model_name, depth_acc in results.items():
        depths = list(depth_acc.keys())
        accs   = [depth_acc[d] for d in depths]
        ax.plot(depths, accs, marker="o", label=model_name)
    ax.set_xlabel("Depth"); ax.set_ylabel("Probe Accuracy (%)")
    ax.set_title("Layer-Wise Feature Probing")
    ax.legend(); ax.grid(True)
    _save(fig, save_path)


def plot_feature_norms(
    norms: dict,   # {model: {depth_label: mean_norm}}
    save_path: str,
):
    fig, ax = plt.subplots(figsize=(9, 5))
    for model_name, depth_norms in norms.items():
        depths = list(depth_norms.keys())
        ns     = [depth_norms[d] for d in depths]
        ax.plot(depths, ns, marker="s", label=model_name)
    ax.set_xlabel("Depth"); ax.set_ylabel("Mean Feature L2 Norm")
    ax.set_title("Feature Norm Statistics Across Layers")
    ax.legend(); ax.grid(True)
    _save(fig, save_path)