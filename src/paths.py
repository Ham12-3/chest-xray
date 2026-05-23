"""Path resolution that works locally and on Kaggle.

Locally, data lives in ./data and outputs in ./artifacts. On Kaggle the data
is read-only under /kaggle/input and outputs must go to /kaggle/working. Two
env vars switch the locations without touching any other code:

  CXR_DATA_DIR        where the images and label CSV live
  CXR_ARTIFACTS_DIR   where manifest, splits, and checkpoints are written
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def data_dir():
    return Path(os.environ.get("CXR_DATA_DIR", ROOT / "data"))


def artifacts_dir():
    d = Path(os.environ.get("CXR_ARTIFACTS_DIR", ROOT / "artifacts"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def splits_dir():
    d = artifacts_dir() / "splits"
    d.mkdir(parents=True, exist_ok=True)
    return d


def manifest_path():
    return artifacts_dir() / "manifest.csv"


def metadata_path():
    return artifacts_dir() / "metadata_present.csv"
