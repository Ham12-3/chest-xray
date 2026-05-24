# Chest X-ray Classifier

A multi-label chest X-ray classifier built with transfer learning on the NIH ChestX-ray14 dataset.

Fine-tunes DenseNet-121 (pretrained on ImageNet) to predict 14 chest findings per image, each as an independent probability. Built as a learning and portfolio project.

---

## Not a medical device

This is a **learning and portfolio project**. It is **NOT** a clinical or diagnostic tool.

The model is not clinically validated and must never be used to diagnose or treat patients. Output from this project, including the API, is for educational purposes only.

The labels come from NLP-mined radiology reports, not from doctors confirming the images. An independent review (Oakden-Rayner, 2017) found the labels often do not match what is visible in the image. There is a real accuracy ceiling because the training targets are noisy.

---

## Results

Trained on the NIH ChestX-ray14 sample (5,606 images, CPU, 3 epochs). Full GPU run on the complete 112k dataset is in progress.

| Finding | AUC |
|---|---|
| Edema | 0.831 |
| Pleural Thickening | 0.822 |
| Emphysema | 0.780 |
| Effusion | 0.778 |
| Fibrosis | 0.773 |
| Consolidation | 0.725 |
| Infiltration | 0.715 |
| Atelectasis | 0.716 |
| Pneumothorax | 0.695 |
| Mass | 0.642 |
| Cardiomegaly | 0.587 |
| Nodule | 0.480 |
| Pneumonia | 0.442 |
| Hernia | n/a (too rare in sample) |
| **Macro average** | **0.691** |

Target from the literature (full dataset, GPU): macro AUC ~0.78 to 0.81.

The pattern matches published work: Effusion and Edema are easier, Nodule and Pneumonia are harder. The four findings in the Infiltration / Consolidation / Atelectasis / Pneumonia group overlap heavily even in the original labels.

---

## Project layout

```
src/
  config.py       load config.yaml
  data.py         dataset class, patient-level split, no-leakage guard
  preprocess.py   transforms and augmentation
  model.py        DenseNet-121 with 14-output head
  train.py        training loop
  evaluate.py     AUC, bootstrap CI, precision/recall
  gradcam.py      Grad-CAM heatmaps
  serve.py        FastAPI inference server
  paths.py        portable path resolution (local vs Kaggle)

scripts/
  prepare_data.py     build image manifest
  make_split.py       patient-level train/val/test split
  smoke_pipeline.py   end-to-end smoke test
  evaluate_test.py    test-set evaluation with bootstrap CI

kaggle/
  run_kaggle.py         full 112k dataset training runner
  run_kaggle_sample.py  sample dataset training runner
```

---

## How to run

### Quick local training (CPU, a few minutes)

```bash
python -m src.train --epochs 3 --batch-size 8 --num-workers 0 \
    --limit-train-batches 80 --limit-val-batches 40
```

### Full training (requires GPU, use Kaggle)

Follow `kaggle/README.md`. Paste `kaggle/run_kaggle.py` into a Kaggle notebook with GPU T4 and the NIH Chest X-rays dataset attached.

### Test-set evaluation (after training)

```bash
python scripts/evaluate_test.py
```

Loads `artifacts/best.pt`, runs on the held-out test split, prints per-finding AUC with 95% bootstrap confidence intervals and precision/recall at val-selected thresholds.

### Run the API server

```bash
pip install fastapi uvicorn python-multipart
python -m src.serve
```

Then POST a chest X-ray image:

```bash
curl -X POST http://localhost:8000/predict \
    -F "file=@your_image.png" | python -m json.tool
```

Response includes 14 probabilities, the top predicted finding, and a Grad-CAM heatmap (base64 PNG) showing which region the model focused on.

---

## Dataset

NIH ChestX-ray14. About 112,120 frontal chest X-ray images from about 30,805 unique patients, 14 findings labelled per image.

Download source: https://www.kaggle.com/datasets/nih-chest-xrays/data

The data folder is gitignored and never committed. The official split lists (`train_val_list.txt`, `test_list.txt`) are patient-disjoint by construction and are used to make results comparable to published work.

---

## The rule that matters most

**Split by patient, never by image.** The same patient must never appear in more than one of train, validation, or test. The code asserts this and raises an error if it is ever violated. Splitting by image instead allows the model to recognise the same patient across splits, making metrics meaningless.

---

## Label noise

The labels were extracted from radiology report text using NLP. They were never confirmed by a clinician reviewing the images directly. An independent radiologist review found many labels do not match what is actually visible. This creates a hard ceiling on how accurate any model trained on this dataset can be, regardless of architecture. Results should be interpreted with this in mind.

---

## Tech stack

Python, PyTorch, DenseNet-121 pretrained on ImageNet, FastAPI, scikit-learn.
