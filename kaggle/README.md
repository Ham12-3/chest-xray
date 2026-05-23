# Running on Kaggle GPU

The real training runs on a free Kaggle GPU notebook. The full NIH dataset is
already hosted on Kaggle, so there is no upload.

## Steps

1. Go to kaggle.com and create a new Notebook.

2. In the notebook settings (right panel):
   - Set Accelerator to GPU (T4 or P100).
   - Set Persistence off is fine.

3. Click "Add Input", search for "NIH Chest X-rays" by nih-chest-xrays, and
   add the full dataset named "data". It mounts at `/kaggle/input/data`.

4. Open `kaggle/run_kaggle.py` from this repo, copy the whole file into one
   notebook cell, set `REPO_URL` to this repo's URL, and run the cell.

That cell will:
   - clone the repo
   - point the code at `/kaggle/input/data` for input and
     `/kaggle/working/artifacts` for output
   - build the image manifest
   - build the patient-level split using the official `train_val_list.txt`
     and `test_list.txt`
   - train DenseNet-121 on the GPU

5. When it finishes, download `best.pt` and the split files from the notebook
   Output tab (`/kaggle/working/artifacts`).

## Notes

- Do NOT run `pip install -r requirements.txt` on Kaggle. Kaggle ships a CUDA
  build of torch. Reinstalling would swap in the CPU build and training would
  crawl. The other dependencies are already present.

- Training time: DenseNet-121 at 224px over the full train_val set is roughly
  10 to 15 minutes per epoch on a T4. The default is 15 epochs with early
  stopping. For a first run you can lower epochs in `config.yaml`, or pass an
  override by editing the last command to:
  `python -m src.train --epochs 6`.

- Kaggle GPU sessions have a time limit (about 9 to 12 hours) and a weekly
  quota (about 30 hours). The schedule above fits comfortably.

- The code auto-detects the GPU and turns on mixed precision. No change needed.
