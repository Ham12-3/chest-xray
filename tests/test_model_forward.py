"""Verify the model and transforms on dummy data.

No dataset and no internet needed. We use pretrained=False here so the check
runs offline. The real training run uses pretrained=True.

Run:  python tests/test_model_forward.py
"""

import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import build_model  # noqa: E402
from src.preprocess import build_transforms  # noqa: E402

IMAGE_CFG = {"size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def test_transforms_shape():
    fake = Image.new("RGB", (1024, 1024))  # mimic a 1024x1024 X-ray
    train_t = build_transforms(IMAGE_CFG, train=True)
    eval_t = build_transforms(IMAGE_CFG, train=False)
    assert train_t(fake).shape == (3, 224, 224)
    assert eval_t(fake).shape == (3, 224, 224)
    print("ok: transforms produce 3x224x224 tensors")


def test_model_forward_and_step():
    model_cfg = {"name": "densenet121", "pretrained": False, "num_classes": 14}
    net = build_model(model_cfg)

    batch = torch.randn(4, 3, 224, 224)  # 4 fake images
    logits = net(batch)
    assert logits.shape == (4, 14), f"expected (4, 14), got {tuple(logits.shape)}"
    print(f"ok: forward pass output shape {tuple(logits.shape)} (logits, no sigmoid)")

    # One training step with the multi-label loss and per-class pos_weight.
    targets = torch.randint(0, 2, (4, 14)).float()
    pos_weight = torch.full((14,), 5.0)  # stand-in for the real class weights
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss = loss_fn(logits, targets)
    loss.backward()
    assert torch.isfinite(loss), "loss is not finite"
    print(f"ok: one BCEWithLogitsLoss step, loss={loss.item():.4f}, backward ran")

    # Inference path: logits to probabilities via sigmoid.
    probs = torch.sigmoid(logits)
    assert ((probs >= 0) & (probs <= 1)).all()
    print("ok: sigmoid gives valid probabilities in [0, 1]")


if __name__ == "__main__":
    test_transforms_shape()
    test_model_forward_and_step()
    print("\nall checks passed")
