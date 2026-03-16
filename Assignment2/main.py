
import argparse
import json
import os
import random
import sys
import time
import numpy as np
import torch

import config
from dataset   import extract_dataset, make_loaders
from scenarios import (
    run_linear_probe,
    run_finetuning_strategies,
    run_few_shot,
    run_corruption_robustness,
    run_layer_probing,
)


# ─── Tee Logger ──────────────────────────────────────────────────────────────

class TeeLogger:
    """
    Mirrors every write to stdout/stderr to both the terminal AND a log file.
    Call TeeLogger(path) once at startup; call .close() at the end.
    """
    def __init__(self, log_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        self._terminal_out = sys.stdout
        self._terminal_err = sys.stderr
        self._log = open(log_path, "w", buffering=1)   # line-buffered
        sys.stdout = self
        sys.stderr = self
        self.log_path = log_path

    def write(self, message):
        self._terminal_out.write(message)
        self._log.write(message)

    def flush(self):
        self._terminal_out.flush()
        self._log.flush()

    def close(self):
        sys.stdout = self._terminal_out
        sys.stderr = self._terminal_err
        self._log.close()
        print(f"\n[logger] Full terminal log saved → {self.log_path}")


# ─── Reproducibility ─────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="GNR638 Assignment 2")
    p.add_argument(
        "--scenarios", nargs="+",
        choices=["4.1", "4.2", "4.3", "4.4", "4.5"],
        default=["4.1", "4.2", "4.3", "4.4", "4.5"],
        help="Which scenarios to run (default: all)",
    )
    p.add_argument(
        "--models", nargs="+",
        choices=["resnet50", "efficientnet_b0", "densenet121",
                 "inception_v3", "convnext_tiny"],
        default=None,
        help="Override model list (default: from config.py)",
    )
    p.add_argument("--data-zip",    default=config.DATA_ZIP)
    p.add_argument("--data-root",   default=config.DATA_ROOT)
    p.add_argument("--skip-extract", action="store_true",
                   help="Skip zip extraction (data already at --data-root)")
    p.add_argument("--results-dir", default=config.RESULTS_DIR)
    p.add_argument("--batch-size",  type=int, default=config.BATCH_SIZE)
    p.add_argument("--epochs-full",    type=int, default=config.EPOCHS_FULL,
                   help="Max epochs for full-data training (default: 30)")
    p.add_argument("--epochs-fewshot", type=int, default=config.EPOCHS_FEWSHOT,
                   help="Max epochs for few-shot training (default: 20)")
    p.add_argument("--seed",        type=int, default=config.SEED)
    p.add_argument("--log-file",    default=None,
                   help="Path to save terminal log (default: <results-dir>/run.log)")
    return p.parse_args()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Apply CLI overrides to config
    if args.models:
        config.MODELS = args.models
    config.RESULTS_DIR   = args.results_dir
    config.BATCH_SIZE    = args.batch_size
    config.EPOCHS_FULL   = args.epochs_full
    config.EPOCHS_FEWSHOT= args.epochs_fewshot
    config.SEED          = args.seed

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINTS, exist_ok=True)

    # ── Start logger (before anything else prints) ────────────────────────────
    log_path = args.log_file or os.path.join(config.RESULTS_DIR, "run.log")
    logger   = TeeLogger(log_path)

    # Log run metadata at the top of the file
    print("=" * 65)
    print("  GNR638 Assignment 2 – Run Log")
    print(f"  Started  : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Log file : {log_path}")
    print("=" * 65)

    set_seed(config.SEED)

    # ── Dataset root ─────────────────────────────────────────────────────────

    if args.skip_extract or os.path.isdir(args.data_root):
        data_root = args.data_root
    else:
        data_root = extract_dataset(args.data_zip, args.data_root)
    config.DATA_ROOT = data_root

    # ── Discover class names ──────────────────────────────────────────────────
    from torchvision import datasets
    from dataset import get_transforms, diagnose_dataset

   
    diagnose_dataset(data_root)

    ds = datasets.ImageFolder(data_root, transform=get_transforms(config.IMAGE_SIZE, "val"))
    class_names = ds.classes

    if len(class_names) < 2:
        raise RuntimeError(
            f"\n[FATAL] Only {len(class_names)} class(es) found at: {data_root}\n"
            f"Expected 30 class subdirectories. Check DATA_ROOT in config.py."
        )

    print(f"\n[main] Dataset   : {len(ds)} images | {len(class_names)} classes  ✓")
    print(f"[main] Models    : {config.MODELS}")
    print(f"[main] Scenarios : {args.scenarios}")
    print(f"[main] Device    : {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"[main] Seed      : {config.SEED}\n")

    # ── Run scenarios ─────────────────────────────────────────────────────────
    all_results = {}
    t_global = time.time()

    try:
        if "4.1" in args.scenarios:
            all_results["4.1_linear_probe"] = run_linear_probe(data_root, class_names)

        if "4.2" in args.scenarios:
            all_results["4.2_finetune"] = run_finetuning_strategies(data_root, class_names)

        if "4.3" in args.scenarios:
            all_results["4.3_few_shot"] = run_few_shot(data_root, class_names)

        if "4.4" in args.scenarios:
            all_results["4.4_corruption"] = run_corruption_robustness(data_root, class_names)

        if "4.5" in args.scenarios:
            all_results["4.5_layer_probe"] = run_layer_probing(data_root, class_names)

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

    finally:
        # ── Save combined JSON ──────────────────────────────────────────────
        combined_path = os.path.join(config.RESULTS_DIR, "all_results.json")
        with open(combined_path, "w") as f:
            json.dump(
                all_results, f, indent=2,
                default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x),
            )

        elapsed = (time.time() - t_global) / 60
        print(f"\n[main] All results saved → {combined_path}")
        print(f"[main] Total time : {elapsed:.1f} min")
        print(f"[main] Finished   : {time.strftime('%Y-%m-%d %H:%M:%S')}")

        logger.close()


if __name__ == "__main__":
    main()