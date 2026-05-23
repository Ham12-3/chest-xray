"""Checks for the data module, run on synthetic data.

We fake a CSV that looks like Data_Entry_2017.csv, then prove:
  1. Findings parse into the right 14 binary columns.
  2. The split is patient-disjoint.
  3. The leakage guard actually raises when patients overlap.

Run it directly:  python tests/test_data_split.py
"""

import sys
from pathlib import Path

import pandas as pd

# Make src importable when run as a plain script.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import (  # noqa: E402
    LABELS,
    add_label_columns,
    assert_no_patient_overlap,
    build_split,
)


def make_synthetic_metadata(n_patients=60, seed=0):
    """Build a fake metadata frame: many patients, several images each."""
    import random

    rng = random.Random(seed)
    rows = []
    for pid in range(1, n_patients + 1):
        n_images = rng.randint(1, 5)  # patients have follow-up scans
        for k in range(n_images):
            # Pick 0 to 3 findings, or No Finding.
            if rng.random() < 0.5:
                finding = "No Finding"
            else:
                picks = rng.sample(LABELS, rng.randint(1, 3))
                finding = "|".join(picks)
            rows.append(
                {
                    "Image Index": f"{pid:08d}_{k:03d}.png",
                    "Finding Labels": finding,
                    "Patient ID": pid,
                }
            )
    return pd.DataFrame(rows)


def test_label_parsing():
    df = pd.DataFrame(
        {
            "Image Index": ["a.png", "b.png", "c.png"],
            "Finding Labels": ["Cardiomegaly|Effusion", "No Finding", "Hernia"],
            "Patient ID": [1, 2, 3],
        }
    )
    df = add_label_columns(df)
    assert df.loc[0, "Cardiomegaly"] == 1
    assert df.loc[0, "Effusion"] == 1
    assert df.loc[0, "Hernia"] == 0
    assert df.loc[1, LABELS].sum() == 0  # No Finding is all zeros
    assert df.loc[2, "Hernia"] == 1
    print("ok: label parsing")


def test_split_is_patient_disjoint():
    df = make_synthetic_metadata(n_patients=60)
    df = add_label_columns(df)

    # Fake official lists: patients 1-48 in train_val, 49-60 in test.
    train_val_images = set(df[df["Patient ID"] <= 48]["Image Index"])
    test_images = set(df[df["Patient ID"] > 48]["Image Index"])

    splits = build_split(
        df,
        val_fraction=0.2,
        seed=42,
        train_val_images=train_val_images,
        test_images=test_images,
    )

    # Every image must be accounted for exactly once.
    total = sum(len(s) for s in splits.values())
    assert total == len(df), f"image count mismatch: {total} vs {len(df)}"

    # No patient may appear in two splits. (Redundant with the guard inside
    # build_split, but we assert here too as a direct check.)
    assert_no_patient_overlap(splits)

    sizes = {name: len(s) for name, s in splits.items()}
    patients = {name: s["Patient ID"].nunique() for name, s in splits.items()}
    print(f"ok: patient-disjoint split  images={sizes}  patients={patients}")


def test_guard_catches_leakage():
    """A deliberately bad split must raise."""
    df = make_synthetic_metadata(n_patients=20)
    df = add_label_columns(df)
    bad = {
        "train": df[df["Patient ID"] <= 10],
        "val": df[df["Patient ID"] >= 9],  # patients 9, 10 overlap on purpose
        "test": df[df["Patient ID"] > 15],
    }
    try:
        assert_no_patient_overlap(bad)
    except ValueError as e:
        print(f"ok: guard caught leakage -> {e}")
        return
    raise AssertionError("guard did not catch the planted leak")


if __name__ == "__main__":
    test_label_parsing()
    test_split_is_patient_disjoint()
    test_guard_catches_leakage()
    print("\nall checks passed")
