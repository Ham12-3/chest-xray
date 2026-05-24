"""Quick inference test on a single image.

Loads best.pt, runs it on one image, prints the 14 probabilities, and saves
a Grad-CAM overlay showing what the model looked at.

Run:
    python scripts/test_inference.py
    python scripts/test_inference.py --image path/to/your_image.png
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image
from src.data import LABELS
from src.gradcam import GradCAM
from src.model import build_model
from src.paths import artifacts_dir, data_dir
from src.preprocess import build_transforms


def find_sample_image():
    for pattern in ["**/*.png", "**/*.jpg"]:
        matches = list(data_dir().glob(pattern))
        if matches:
            return matches[0]
    return None


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="Path to a chest X-ray image")
    ap.add_argument("--output", default="gradcam_output.png", help="Where to save the heatmap overlay")
    return ap.parse_args()


def main():
    args = parse_args()

    ckpt_path = artifacts_dir() / "best.pt"
    if not ckpt_path.exists():
        print(f"No checkpoint found at {ckpt_path}")
        print("Run training first: python -m src.train --epochs 3 --batch-size 8 --num-workers 0 --limit-train-batches 80 --limit-val-batches 40")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    print(f"Val macro AUC at save: {ckpt.get('val_macro_auc', 0):.4f}")

    model_cfg = cfg.get("model", {"name": "densenet121", "pretrained": False, "num_classes": 14})
    model = build_model(model_cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    image_cfg = cfg.get("image", {"size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]})
    transform = build_transforms(image_cfg, train=False)

    image_path = Path(args.image) if args.image else find_sample_image()
    if image_path is None or not image_path.exists():
        print("No image found. Pass --image path/to/image.png")
        sys.exit(1)

    print(f"\nImage: {image_path.name}")
    pil_image = Image.open(image_path).convert("RGB")
    tensor = transform(pil_image).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
    probs = torch.sigmoid(logits)[0].cpu()

    print("\nPredictions:")
    print(f"  {'Finding':22s} {'Probability':>11s}")
    print(f"  {'-'*22} {'-'*11}")
    for label, p in sorted(zip(LABELS, probs.tolist()), key=lambda x: -x[1]):
        bar = "#" * int(p * 20)
        print(f"  {label:22s} {p:11.3f}  {bar}")

    top_idx = int(probs.argmax())
    print(f"\nTop finding: {LABELS[top_idx]} ({probs[top_idx]:.3f})")

    print(f"\nGenerating Grad-CAM for '{LABELS[top_idx]}'...")
    cam = GradCAM(model)
    heatmap = cam.generate(tensor, top_idx)
    overlay = cam.overlay(pil_image.resize((224, 224)), heatmap)
    overlay.save(args.output)
    print(f"Grad-CAM overlay saved to: {args.output}")
    print("\nOpen that file to see which region the model focused on.")


if __name__ == "__main__":
    main()
