"""Prepare the downloaded data.

Works no matter how the download laid out the folders, and works the same
locally or on Kaggle (paths come from src/paths.py). It:
  1. Scans the data dir recursively for every PNG and records where each is.
  2. Finds the labels CSV (handles the full set and the sample naming).
  3. Keeps only metadata rows whose image is actually on disk, so a partial
     download (the sample, or one tarball) just works.
  4. Saves a manifest and the filtered metadata to the artifacts dir.
  5. Prints counts and label prevalence so we can sanity check what landed.

Run:  python scripts/prepare_data.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import IMAGE_COL, LABELS, PATIENT_COL, add_label_columns  # noqa: E402
from src.paths import data_dir, manifest_path, metadata_path  # noqa: E402

# Known names for the labels file, in order of preference.
LABEL_CSV_NAMES = ["Data_Entry_2017.csv", "Data_Entry_2017_v2020.csv", "sample_labels.csv"]


def find_images(root):
    """Map every PNG filename to its full path on disk."""
    return {p.name: p for p in root.rglob("*.png")}


def find_label_csv(root):
    """Locate the labels CSV by known name, then by column signature."""
    for name in LABEL_CSV_NAMES:
        hits = list(root.rglob(name))
        if hits:
            return hits[0]
    for csv in root.rglob("*.csv"):
        try:
            cols = pd.read_csv(csv, nrows=1).columns
            if "Finding Labels" in cols:
                return csv
        except Exception:
            continue
    return None


def main():
    root = data_dir()
    if not root.exists():
        sys.exit(f"data dir does not exist: {root}")

    manifest = find_images(root)
    print(f"images found on disk: {len(manifest)}")

    csv = find_label_csv(root)
    if csv is None:
        sys.exit(f"could not find a labels CSV under {root}")
    print(f"labels csv: {csv}")

    df = pd.read_csv(csv)
    print(f"rows in csv: {len(df)}")

    df = add_label_columns(df)
    present = df[df[IMAGE_COL].isin(manifest.keys())].copy()
    print(f"rows with image present: {len(present)}")
    print(f"unique patients present: {present[PATIENT_COL].nunique()}")

    pd.DataFrame(
        {"Image Index": list(manifest.keys()), "path": [str(p) for p in manifest.values()]}
    ).to_csv(manifest_path(), index=False)
    present.to_csv(metadata_path(), index=False)
    print(f"\nwrote {manifest_path()}")
    print(f"wrote {metadata_path()}")

    print("\nlabel prevalence (present images):")
    for label in LABELS:
        count = int(present[label].sum())
        pct = present[label].mean() * 100 if len(present) else 0
        print(f"  {label:20s} {count:6d}  ({pct:4.1f}%)")
    no_finding = (present[LABELS].sum(axis=1) == 0).sum()
    print(f"  {'(no finding)':20s} {no_finding:6d}  ({no_finding / len(present) * 100:4.1f}%)")


if __name__ == "__main__":
    main()
