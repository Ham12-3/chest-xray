# PROGRESS.md

Progress tracker for the chest X-ray classifier.

This file is the single place to see what is done and what is left. Update the boxes as we go.

Legend:
- [x] done
- [~] in progress
- [ ] not started

## Where we are right now

Spec and research are done. Skeleton, config, data-and-split, preprocessing, and the model are built.

The split guard, preprocessing, and the model forward and loss step are all tested and passing.

Sample data downloaded, prepared, and split. Training loop and metrics built and smoke-tested on CPU.

Code is pushed to GitHub at https://github.com/Ham12-3/chest-xray (branch main).

Phase 6 is complete. Evaluation script (scripts/evaluate_test.py) is built and pushed.

Next: run the real training on a Kaggle GPU notebook using kaggle/run_kaggle.py, then run scripts/evaluate_test.py on the downloaded best.pt to get final test-set numbers.

## Phase 0: Setup and research

- [x] Project spec written in CLAUDE.md
- [x] CLAUDE.md improved with dataset facts, hard rules detail, known traps, target AUCs, and a baseline recipe
- [x] gstack installed
- [x] Research brief done: benchmarks, dataset pitfalls, training recipe, tooling
- [x] git init for the repo (on main)
- [x] requirements.txt
- [x] config file seeded with the baseline recipe (config.yaml)
- [x] README with project intro and the not-a-diagnostic-tool disclaimer
- [x] .gitignore that excludes the data folder

## Phase 1: Get the data

- [x] Download the data. Kaggle client set up and authenticated. The official sample is downloaded: 5,606 images, 4,230 patients.
- [x] Helper scripts written: scripts/prepare_data.py (image manifest plus filter metadata to present images) and scripts/make_split.py
- [x] Ran prepare_data.py. Counts and label prevalence match the known NIH distribution.
- [ ] Document the download source in the README
- [ ] Later: get the full set (free disk to ~95GB, or use an external drive)

## Phase 2: Data module and patient-level split

This phase enforces hard rule 1. Do not skip the overlap check.

- [x] Load the CSV and parse the 14 findings into a multi-label vector (add_label_columns)
- [x] Build the patient-level split using the official lists (build_split)
- [x] Carve a validation set out of train_val, grouped by patient
- [x] Assert that the patient ID sets of train, val, and test do not overlap (assert_no_patient_overlap, tested)
- [x] Save the split to files so runs are reproducible (save_split)
- [x] Dataset class returns an image plus a 14-dim label vector (ChestXrayDataset, verified on real images)
- [x] Real split built on the sample: train 4016, val 749, test 841, no patient overlap
- [x] Full-path smoke test passed (scripts/smoke_pipeline.py): real PNG to transforms to dataloader to model logits

## Phase 3: Preprocessing and augmentation

- [x] Resize to 224x224 (src/preprocess.py)
- [x] Grayscale replicated to 3 channels (Dataset converts to RGB)
- [x] Normalize with ImageNet mean and std
- [x] Light augmentation: small rotation, mild zoom or crop, slight brightness and contrast
- [x] Horizontal flip decided: OFF by default (it swaps left and right). No vertical flip.
- [x] Wire up the DataLoader (done in smoke_pipeline.py, will reuse in training)

## Phase 4: Model

- [x] DenseNet-121 pretrained on ImageNet (src/model.py)
- [x] Replace the classifier with a 14-unit linear head
- [x] One independent sigmoid per finding, no softmax (hard rule 2). Model returns logits; sigmoid applied at inference.

## Phase 5: Training loop

- [x] BCEWithLogitsLoss with per-class pos_weight (src/train.py, hard rule 4)
- [x] Adam optimizer, learning rate about 1e-4
- [x] Reduce LR on plateau of validation macro AUC
- [x] Mixed precision (AMP), turns on only on CUDA
- [ ] Warm up the head, then fine-tune the whole network (optional refinement, currently fine-tunes from the start)
- [x] Save the best checkpoint by validation macro AUC
- [x] Early stopping
- [x] Everything driven by the config file, with a logged seed
- [x] Tiny CPU smoke passed: loop trains, scheduler steps, checkpoint saves, AUC report prints

## Phase 6: Evaluation and metrics

This phase enforces hard rule 3. Lead with AUC, show the full table.

- [x] AUC-ROC per finding (src/evaluate.py)
- [x] Macro-averaged AUC
- [x] Bootstrap confidence intervals (function built, run during full test eval)
- [x] Precision and recall (function built; thresholds selected on val via max-F1 sweep)
- [x] Print the per-finding table, not just the macro average
- [x] Standalone test-set evaluation script (scripts/evaluate_test.py)

## Phase 7: Baseline run and iterate

- [ ] Smoke run end to end on the subset
- [ ] Full baseline training run
- [ ] Compare results to targets: mean AUC about 0.78 to 0.81
- [ ] Sanity check the per-finding spread (easy vs hard findings)
- [ ] Note any findings that look too good, which can mean leakage

## Phase 8: Explainability and serving (optional, stretch)

This goes beyond the original five modules. Nice for a portfolio.

- [ ] Grad-CAM heatmaps to show where the model looks
- [ ] FastAPI endpoint that takes an image and returns the 14 probabilities
- [ ] Clear disclaimer in the API response and UI: not a diagnostic tool

## Phase 9: Docs and portfolio polish

- [ ] README with results, the per-finding AUC table, and example predictions
- [ ] State the label-noise ceiling and why labels are not ground truth
- [ ] State plainly that this is a learning project, not clinically validated

## Open questions to settle before or during the build

- [ ] Where will training run: local GPU, Colab, or a cloud instance?
- [ ] Start at 224px, or go higher later for small findings like nodules?
- [ ] Use Weights and Biases for tracking, or TensorBoard?
