"""Run training on a Kaggle GPU notebook.

How to use (see kaggle/README.md for the full walkthrough):
  1. Create a new Kaggle notebook, set Accelerator to GPU.
  2. Add Data: search "NIH Chest X-rays" by nih-chest-xrays and attach the
     full "data" dataset. It mounts at /kaggle/input/data.
  3. Paste this whole file into one cell, set REPO_URL below, and run.

Do NOT pip install -r requirements.txt on Kaggle. Kaggle already ships a CUDA
build of torch, and reinstalling would replace it with the CPU build. The other
deps (pandas, sklearn, pyyaml, pillow) are preinstalled too.
"""

import os
import subprocess
import sys

# Filled in after the repo is pushed to GitHub.
REPO_URL = "https://github.com/REPLACE_ME/chest-xray.git"
REPO_DIR = "chest-xray"

# Kaggle paths: input is read-only, working is writable.
DATA_DIR = "/kaggle/input/data"
ARTIFACTS_DIR = "/kaggle/working/artifacts"


def sh(cmd, **kw):
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kw)


def main():
    if not os.path.exists(REPO_DIR):
        sh(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR])
    os.chdir(REPO_DIR)

    env = dict(os.environ)
    env["CXR_DATA_DIR"] = DATA_DIR
    env["CXR_ARTIFACTS_DIR"] = ARTIFACTS_DIR

    # Build the manifest, then the patient-level split (uses the official
    # lists that ship with the full dataset), then train on the GPU.
    sh([sys.executable, "scripts/prepare_data.py"], env=env)
    sh([sys.executable, "scripts/make_split.py"], env=env)
    sh([sys.executable, "-m", "src.train"], env=env)

    print("\nDone. Checkpoint and outputs are in /kaggle/working/artifacts.")
    print("Download best.pt from the notebook Output tab.")


if __name__ == "__main__":
    main()
