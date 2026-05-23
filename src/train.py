"""Training loop.

Reads the config, builds the split-based loaders, the model, and the loss with
per-class pos_weight, then trains. It tracks validation macro AUC, reduces the
learning rate on plateau, saves the best checkpoint, and early-stops.

Device aware. It uses CUDA automatically when present (Kaggle GPU) and falls
back to CPU. Mixed precision turns on only on CUDA.

CLI overrides exist so we can run a fast smoke before a real run:
  python -m src.train --epochs 1 --limit-train-batches 4 --limit-val-batches 4 \
      --batch-size 8 --num-workers 0 --no-pretrained
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.data import IMAGE_COL, LABELS, ChestXrayDataset, _read_image_list  # noqa: E402
from src.evaluate import format_report, macro_auc, per_finding_auc  # noqa: E402
from src.model import build_model  # noqa: E402
from src.paths import artifacts_dir, manifest_path, metadata_path, splits_dir  # noqa: E402
from src.preprocess import build_transforms  # noqa: E402


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split_df(meta, split_dir, name):
    images = _read_image_list(split_dir / f"{name}.txt")
    return meta[meta[IMAGE_COL].isin(images)].reset_index(drop=True)


def compute_pos_weight(train_df, clip=50.0):
    """Per-class neg/pos ratio, clipped so rare classes do not explode loss."""
    pos = train_df[LABELS].sum(axis=0).values.astype("float32")
    pos = np.clip(pos, 1, None)  # guard against zero positives
    neg = len(train_df) - pos
    weight = np.clip(neg / pos, None, clip)
    return torch.tensor(weight, dtype=torch.float32)


@torch.no_grad()
def evaluate_loader(model, loader, device, limit_batches=None):
    model.eval()
    ys, ps = [], []
    for i, (x, y) in enumerate(loader):
        if limit_batches and i >= limit_batches:
            break
        logits = model(x.to(device))
        ps.append(torch.sigmoid(logits).cpu().numpy())
        ys.append(y.numpy())
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)
    return per_finding_auc(y_true, y_prob)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--limit-train-batches", type=int, default=None)
    ap.add_argument("--limit-val-batches", type=int, default=None)
    ap.add_argument("--no-pretrained", action="store_true", help="skip ImageNet weights (smoke)")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = args.epochs or cfg["train"]["epochs"]
    batch_size = args.batch_size or cfg["train"]["batch_size"]
    num_workers = args.num_workers if args.num_workers is not None else cfg["train"]["num_workers"]
    print(f"device: {device} | epochs: {epochs} | batch_size: {batch_size}")

    # Data
    manifest = pd.read_csv(manifest_path())
    path_map = dict(zip(manifest["Image Index"], manifest["path"]))
    meta = pd.read_csv(metadata_path())
    split_dir = splits_dir()
    train_df = load_split_df(meta, split_dir, "train")
    val_df = load_split_df(meta, split_dir, "val")
    print(f"train rows: {len(train_df)} | val rows: {len(val_df)}")

    train_ds = ChestXrayDataset(train_df, path_map=path_map, transform=build_transforms(cfg["image"], train=True))
    val_ds = ChestXrayDataset(val_df, path_map=path_map, transform=build_transforms(cfg["image"], train=False))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Model, loss, optimizer
    model_cfg = dict(cfg["model"])
    if args.no_pretrained:
        model_cfg["pretrained"] = False
    model = build_model(model_cfg).to(device)

    pos_weight = compute_pos_weight(train_df).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg["train"]["lr_factor"], patience=cfg["train"]["lr_patience"]
    )

    use_amp = device.type == "cuda" and cfg["train"]["mixed_precision"]
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    ckpt_dir = artifacts_dir()
    best_auc = -1.0
    bad_epochs = 0
    patience = cfg["train"]["early_stopping_patience"]
    aucs = {}

    for epoch in range(1, epochs + 1):
        model.train()
        running, n_batches = 0.0, 0
        t0 = time.time()
        for i, (x, y) in enumerate(train_loader):
            if args.limit_train_batches and i >= args.limit_train_batches:
                break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            if use_amp:
                with torch.autocast(device_type="cuda"):
                    loss = loss_fn(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = loss_fn(model(x), y)
                loss.backward()
                optimizer.step()
            running += loss.item()
            n_batches += 1

        train_loss = running / max(n_batches, 1)
        aucs = evaluate_loader(model, val_loader, device, args.limit_val_batches)
        val_auc = macro_auc(aucs)
        scheduler.step(val_auc)
        print(f"epoch {epoch}: train_loss={train_loss:.4f} val_macro_auc={val_auc:.4f} ({time.time() - t0:.0f}s)")

        if val_auc > best_auc:
            best_auc = val_auc
            bad_epochs = 0
            torch.save(
                {"model": model.state_dict(), "labels": LABELS, "val_macro_auc": val_auc, "config": cfg},
                ckpt_dir / "best.pt",
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"early stop at epoch {epoch}")
                break

    print(f"\nbest val macro AUC: {best_auc:.4f}")
    print("\nper-finding AUC (last validation):")
    print(format_report(aucs))


if __name__ == "__main__":
    main()
