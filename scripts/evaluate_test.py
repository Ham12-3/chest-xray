"""Standalone test-set evaluation.

Run this after training to get final numbers from best.pt.

Usage:
  python scripts/evaluate_test.py
  python scripts/evaluate_test.py --checkpoint path/to/best.pt

What it does:
  1. Loads best.pt (model weights, labels, config).
  2. Runs inference on the test split.
  3. Picks per-finding probability thresholds on the validation split
     (threshold that maximises F1 for each finding), never on test.
  4. Prints per-finding AUC with 95% bootstrap CI, then precision and recall.

The AUC numbers are the main result (hard rule 3). Precision and recall are
secondary because they depend on the threshold choice.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.data import IMAGE_COL, LABELS, ChestXrayDataset, _read_image_list  # noqa: E402
from src.evaluate import (  # noqa: E402
    bootstrap_macro_auc,
    format_report,
    macro_auc,
    per_finding_auc,
    precision_recall_at,
)
from src.model import build_model  # noqa: E402
from src.paths import artifacts_dir, manifest_path, metadata_path, splits_dir  # noqa: E402
from src.preprocess import build_transforms  # noqa: E402


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        logits = model(x.to(device))
        ps.append(torch.sigmoid(logits).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def best_f1_thresholds(y_true, y_prob, grid=50):
    """Per-finding threshold that maximises F1 on a held-out set (val).

    Sweeps grid evenly-spaced thresholds from 0.05 to 0.95 and picks the
    one with the highest F1 for each finding. Falls back to 0.5 when a
    finding has no positives (AUC is undefined and any threshold gives F1=0).
    """
    candidates = np.linspace(0.05, 0.95, grid)
    thresholds = np.full(len(LABELS), 0.5)
    for i in range(len(LABELS)):
        if y_true[:, i].sum() == 0:
            continue
        best_t, best_f1 = 0.5, -1.0
        for t in candidates:
            pred = (y_prob[:, i] >= t).astype(int)
            tp = ((pred == 1) & (y_true[:, i] == 1)).sum()
            fp = ((pred == 1) & (y_true[:, i] == 0)).sum()
            fn = ((pred == 0) & (y_true[:, i] == 1)).sum()
            denom = 2 * tp + fp + fn
            f1 = (2 * tp / denom) if denom > 0 else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[i] = best_t
    return thresholds


def load_split_loader(meta, path_map, split_dir, name, cfg, batch_size, num_workers):
    images = _read_image_list(split_dir / f"{name}.txt")
    df = meta[meta[IMAGE_COL].isin(images)].reset_index(drop=True)
    ds = ChestXrayDataset(df, path_map=path_map, transform=build_transforms(cfg["image"], train=False))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f"  {name}: {len(df)} images")
    return loader


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="path to best.pt (default: artifacts_dir/best.pt)")
    ap.add_argument("--bootstrap", type=int, default=1000, help="bootstrap samples for CI (0 to skip)")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    return ap.parse_args()


def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint) if args.checkpoint else artifacts_dir() / "best.pt"
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        print("Run training first: python -m src.train")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config") or load_config(ROOT / "config.yaml")
    val_auc_at_save = ckpt.get("val_macro_auc", float("nan"))
    print(f"  val macro AUC at checkpoint: {val_auc_at_save:.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    manifest = pd.read_csv(manifest_path())
    path_map = dict(zip(manifest["Image Index"], manifest["path"]))
    meta = pd.read_csv(metadata_path())
    split_dir = splits_dir()

    print("\nLoading splits:")
    val_loader = load_split_loader(meta, path_map, split_dir, "val", cfg, args.batch_size, args.num_workers)
    test_loader = load_split_loader(meta, path_map, split_dir, "test", cfg, args.batch_size, args.num_workers)

    print("\nRunning inference on val (for threshold selection)...")
    val_true, val_prob = run_inference(model, val_loader, device)

    print("Running inference on test...")
    test_true, test_prob = run_inference(model, test_loader, device)

    # AUC on test set — the headline result.
    test_aucs = per_finding_auc(test_true, test_prob)
    test_macro = macro_auc(test_aucs)

    print(f"\n{'='*45}")
    print(f"  Test macro AUC: {test_macro:.4f}")
    print(f"{'='*45}")
    print(format_report(test_aucs))

    if args.bootstrap > 0:
        print(f"\nBootstrap CI ({args.bootstrap} samples)...")
        lo, hi = bootstrap_macro_auc(test_true, test_prob, n_samples=args.bootstrap)
        print(f"  macro AUC 95% CI: [{lo:.4f}, {hi:.4f}]")

    # Threshold selection on val, precision/recall on test.
    print("\nSelecting thresholds on val (max F1 per finding)...")
    thresholds = best_f1_thresholds(val_true, val_prob)
    pr = precision_recall_at(test_true, test_prob, thresholds)

    print("\n  Per-finding precision and recall at val-selected thresholds:")
    header = f"  {'finding':20s} {'thr':>5s} {'prec':>6s} {'rec':>6s}"
    print(header)
    for i, label in enumerate(LABELS):
        p = pr[label]["precision"]
        r = pr[label]["recall"]
        t = thresholds[i]
        print(f"  {label:20s} {t:5.2f} {p:6.3f} {r:6.3f}")

    print(f"\nResults based on: {ckpt_path}")


if __name__ == "__main__":
    main()
