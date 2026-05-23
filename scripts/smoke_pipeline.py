"""Smoke test the full data path on real images.

Loads the manifest and metadata, builds the Dataset and DataLoader with the
real transforms, pulls one batch of actual X-rays, and runs it through the
model. Confirms images on disk flow all the way to model logits.

Uses pretrained=False to stay fast and offline. The real run uses pretrained.

Run:  python scripts/smoke_pipeline.py
"""

import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.data import IMAGE_COL, ChestXrayDataset, _read_image_list  # noqa: E402
from src.model import build_model  # noqa: E402
from src.paths import manifest_path, metadata_path, splits_dir  # noqa: E402
from src.preprocess import build_transforms  # noqa: E402


def main():
    cfg = load_config(ROOT / "config.yaml")

    manifest = pd.read_csv(manifest_path())
    path_map = dict(zip(manifest["Image Index"], manifest["path"]))

    meta = pd.read_csv(metadata_path())
    split_dir = splits_dir()
    train_images = _read_image_list(split_dir / "train.txt")
    train_df = meta[meta[IMAGE_COL].isin(train_images)]
    print(f"train rows: {len(train_df)}")

    transform = build_transforms(cfg["image"], train=True)
    dataset = ChestXrayDataset(train_df, path_map=path_map, transform=transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

    images, labels = next(iter(loader))
    print(f"batch images: {tuple(images.shape)}  labels: {tuple(labels.shape)}")
    assert images.shape == (8, 3, 224, 224)
    assert labels.shape == (8, 14)

    net = build_model({"name": "densenet121", "pretrained": False, "num_classes": 14})
    net.eval()
    with torch.no_grad():
        logits = net(images)
    print(f"model logits: {tuple(logits.shape)}")
    assert logits.shape == (8, 14)

    print("ok: real images flow through transforms -> dataset -> dataloader -> model")


if __name__ == "__main__":
    main()
