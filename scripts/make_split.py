"""Build and save the real train/val/test split.

Reads the filtered metadata from prepare_data.py, uses the official split
lists if they are found under the data dir, otherwise falls back to a
patient-level split. The no-leakage guard runs inside build_split and stops
on any overlap. Works locally or on Kaggle (paths from src/paths.py).

Run:  python scripts/make_split.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.data import PATIENT_COL, _read_image_list, build_split, save_split  # noqa: E402
from src.paths import data_dir, metadata_path, splits_dir  # noqa: E402


def find_official_lists(root):
    train_val = list(root.rglob("train_val_list.txt"))
    test = list(root.rglob("test_list.txt"))
    return (train_val[0] if train_val else None, test[0] if test else None)


def main():
    cfg = load_config(ROOT / "config.yaml")

    meta_file = metadata_path()
    if not meta_file.exists():
        sys.exit("run scripts/prepare_data.py first")
    df = pd.read_csv(meta_file)

    train_val_file, test_file = find_official_lists(data_dir())
    if train_val_file and test_file:
        train_val_images = _read_image_list(train_val_file)
        test_images = _read_image_list(test_file)
        print(f"using the official split lists:\n  {train_val_file}\n  {test_file}")
    else:
        train_val_images = test_images = None
        print("official lists not found, using the patient-level fallback split")

    splits = build_split(
        df,
        val_fraction=cfg["data"]["val_fraction"],
        seed=cfg["seed"],
        train_val_images=train_val_images,
        test_images=test_images,
    )

    save_split(splits, splits_dir())

    print()
    for name, s in splits.items():
        print(f"{name:5s} images={len(s):6d}  patients={s[PATIENT_COL].nunique():5d}")
    print("\nno patient overlap (asserted inside build_split)")
    print(f"split files written to {splits_dir()}")


if __name__ == "__main__":
    main()
