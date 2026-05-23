"""Data loading and the patient-level split.

This module does three jobs:

1. Read the NIH metadata CSV and turn the pipe-separated findings into
   14 binary label columns.
2. Build train, validation, and test splits at the PATIENT level, never
   by image, then assert the patient sets do not overlap.
3. Provide a Dataset that returns an image plus its 14-dim label vector.

Hard rule 1 (no patient leakage) is enforced by assert_no_patient_overlap.
"""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset

# The 14 findings, in a fixed order. This order defines the label vector,
# so it must never change once a model is trained.
LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

# Column names in Data_Entry_2017.csv that we rely on.
IMAGE_COL = "Image Index"
FINDING_COL = "Finding Labels"
PATIENT_COL = "Patient ID"


def add_label_columns(df):
    """Turn the pipe-separated Finding Labels into 14 binary columns.

    "Cardiomegaly|Effusion" becomes 1 in those two columns.
    "No Finding" becomes all zeros.
    """
    findings = df[FINDING_COL].str.split("|")
    for label in LABELS:
        df[label] = findings.apply(lambda fs: int(label in fs))
    return df


def load_metadata(csv_path):
    """Read the CSV and add the binary label columns."""
    df = pd.read_csv(csv_path)
    df = add_label_columns(df)
    return df


def _read_image_list(path):
    """Read a split list file (one image filename per line)."""
    with open(path, "r") as f:
        return {line.strip() for line in f if line.strip()}


def assert_no_patient_overlap(splits):
    """Stop hard if any patient appears in more than one split.

    This is the guard for hard rule 1. A silent leak here would make every
    later metric meaningless, so we fail loudly instead.
    """
    ids = {name: set(df[PATIENT_COL]) for name, df in splits.items()}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = ids[a] & ids[b]
        if overlap:
            sample = list(overlap)[:5]
            raise ValueError(
                f"Patient leakage between {a} and {b}: "
                f"{len(overlap)} shared patients, e.g. {sample}"
            )


def build_split(df, val_fraction, seed, train_val_images=None, test_images=None):
    """Split df into train, val, and test, grouped by patient.

    If the official image lists are given, use them for the train_val vs test
    boundary. Otherwise fall back to a patient-level GroupShuffleSplit.

    Validation is always carved out of train_val by patient, so a whole
    patient stays on one side.
    """
    if train_val_images is not None and test_images is not None:
        train_val_df = df[df[IMAGE_COL].isin(train_val_images)].copy()
        test_df = df[df[IMAGE_COL].isin(test_images)].copy()
    else:
        # No official lists. Hold out 15 percent of patients as test.
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        tv_idx, te_idx = next(gss.split(df, groups=df[PATIENT_COL]))
        train_val_df = df.iloc[tv_idx].copy()
        test_df = df.iloc[te_idx].copy()

    # Carve validation out of train_val, grouped by patient.
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    tr_idx, va_idx = next(gss_val.split(train_val_df, groups=train_val_df[PATIENT_COL]))
    train_df = train_val_df.iloc[tr_idx].copy()
    val_df = train_val_df.iloc[va_idx].copy()

    splits = {"train": train_df, "val": val_df, "test": test_df}
    assert_no_patient_overlap(splits)
    return splits


def save_split(splits, split_dir):
    """Save each split as a plain image list, so runs are reproducible."""
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        df[IMAGE_COL].to_csv(split_dir / f"{name}.txt", index=False, header=False)


class ChestXrayDataset(Dataset):
    """Returns (image, label_vector) for one row of the metadata.

    Image paths resolve one of two ways:
      - path_map: a dict of Image Index to full path. Use this when images
        are scattered across folders (the Kaggle layout). Built by
        scripts/prepare_data.py.
      - image_dir: a single flat folder. Used when every image sits together.

    The transform comes from the preprocessing module (built in the next
    phase). Until then this class still works with any callable transform.
    """

    def __init__(self, df, image_dir=None, path_map=None, transform=None):
        if image_dir is None and path_map is None:
            raise ValueError("give either image_dir or path_map")
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir) if image_dir is not None else None
        self.path_map = path_map
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _resolve(self, image_index):
        if self.path_map is not None:
            return self.path_map[image_index]
        return self.image_dir / image_index

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self._resolve(row[IMAGE_COL])
        image = Image.open(image_path).convert("RGB")  # grayscale to 3 channels
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(row[LABELS].values.astype("float32"))
        return image, label
