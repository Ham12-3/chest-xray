# CLAUDE.md

## Project

A multi-label chest X-ray classifier built with transfer learning.

This is a learning and portfolio project. It is NOT a clinical or diagnostic tool. Any output, README, or demo must state this clearly. Never describe the model as able to diagnose patients.

## Dataset

NIH ChestX-ray14.

- About 112,120 frontal chest X-ray images from about 30,805 unique patients.
- Images are 1024x1024 PNG files. Labels live in a CSV (Data_Entry_2017.csv).
- 14 finding labels. An image can have several findings at once, so this is a multi-label problem, not single-class.
- "No Finding" is the all-zero case, not a 15th label. About half the images have no finding.
- Patients have several images each (follow-up scans). This is exactly why you must split by patient, not by image.

The dataset ships official split lists, train_val_list.txt and test_list.txt. They are already split by patient. Prefer them.

The 14 findings:
Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural_Thickening, Hernia.

### Labels are noisy

The labels were auto-extracted from radiology reports with NLP. They were not confirmed by a doctor looking at the images.

The reports themselves were never released. A radiologist review (Oakden-Rayner, 2017) found the labels often do not match what is actually visible in the image.

So there is a real accuracy ceiling. Do not treat the labels as ground truth in writeups. Say this plainly in the README and explain why.

## Hard rules

These are non-negotiable. Flag it to me if any task would break one.

1. Split the data by patient, never by image. The same patient must never appear in more than one of train, validation, or test. The CSV has a patient ID column. Group by it before splitting. If you split by image, the model learns to recognise patients and the metrics become meaningless.

   Prefer the official train_val_list.txt and test_list.txt. They are patient-disjoint and make results comparable to published work. Carve validation out of train_val, again grouped by patient. Add a check in code that asserts the patient ID sets of train, val, and test do not overlap.

2. Multi-label setup. The model outputs one independent sigmoid probability per finding. Use binary cross-entropy per label. Do NOT use a single softmax across the 14 classes. BCEWithLogitsLoss is the standard choice.

3. Main metric is AUC-ROC per finding, plus macro-averaged AUC. Accuracy is misleading here because most images are normal and many findings are rare. Report precision and recall too, but lead with AUC.

   Always show the per-finding table, not just the macro average, because the spread between findings is wide. Rare classes give noisy AUC, so report bootstrap confidence intervals. Pick any precision or recall thresholds on validation, never on the test set.

4. Handle class imbalance. Most images have no finding and some findings are rare (Hernia is well under one percent, Pneumonia about one percent). Account for this in the loss (for example weighted BCE) or in sampling.

   A simple, robust option is BCE with a per-class pos_weight, roughly num_negatives / num_positives for that label, clipped so rare classes do not explode the loss. Weighted sampling is awkward in multi-label, because one image carries several labels.

## Known traps

- Ambiguous label group. Infiltration, Consolidation, Atelectasis, and Pneumonia overlap heavily. Even radiologists struggle to separate them here. Expect low AUC on these. Do not burn time chasing them.

- View Position shortcut. The AP vs PA column correlates with how sick the patient is, because portable AP films come from sicker, bed-bound patients. A model can learn the view instead of the disease. Do not feed view position as a feature.

- Horizontal flip swaps left and right. The heart sits on the left, so flipping can confuse Cardiomegaly and laterality. Decide on purpose if you use it. Never use vertical flip.

## What good looks like

Rough targets from the literature (CheXNet, Wang et al.). Approximate, not a hard bar. Read per-finding, not just the average.

- Clean DenseNet-121 at 224px on the patient-level split: mean AUC about 0.78 to 0.81.
- Easier findings (Cardiomegaly, Effusion, Edema, Emphysema, Pneumothorax, Hernia): about 0.85 to 0.92.
- Harder findings (Infiltration, Nodule, Mass, Pneumonia, Consolidation): about 0.68 to 0.78.

## Tech stack

- Python, PyTorch.
- DenseNet-121 pretrained on ImageNet as the baseline model, fine-tuned for 14-output multi-label.
- All hyperparameters go in a config file, not hardcoded in scripts.

## Baseline recipe

Starting defaults. All of these live in the config file and are meant to be tuned later.

- Input 224x224. Grayscale replicated to 3 channels. Normalize with ImageNet mean and std, since we use pretrained weights.
- Loss: BCEWithLogitsLoss with per-class pos_weight (see hard rule 4).
- Optimizer Adam, learning rate about 1e-4. Reduce LR on plateau of validation macro AUC. Use mixed precision (AMP).
- Light augmentation only: small rotation, mild zoom or crop, slight brightness and contrast. Nothing that destroys subtle findings.
- Fine-tune the whole network. You can warm up the new head first.
- Save the best checkpoint by validation macro AUC. Use early stopping.

Bigger input (320 or 448) helps small findings like nodules but costs compute. Start at 224 and raise it later if needed.

## Project structure

Keep these as separate modules. Each has one job.

- data loading and the patient-level split
- preprocessing and augmentation
- model definition
- training loop
- evaluation and metrics
- config file for hyperparameters

Two extras that pay off:

- Save the split to files (the image or patient lists per fold) so runs are reproducible and the no-overlap check is easy to prove.
- Keep a small smoke config that runs on a few hundred images, for fast iteration before any full run.

## How I want you to work

- Explain your reasoning before writing significant code, especially for the data split, the loss, and the metrics. I am learning the system, not just collecting code.
- Work in small steps. Build and run one module before moving to the next. Do not dump the whole system at once.
- Run code on a small subset of data first to catch errors fast, before any full run.
- When something breaks, read the error, fix it, and rerun. Tell me what was wrong.

## Writing style for any docs or comments

- Plain, simple English. Short sentences.
- No emdashes. Use commas or periods.
- Well-spaced formatting with line breaks between separate ideas. No dense paragraphs.

## gstack

gstack is installed. Use it for browsing and for its workflow skills.

This assumes each teammate has gstack installed on their own machine. If a skill is missing, run the gstack setup first.

### Web browsing

Use the /browse skill from gstack for all web browsing.

Never use the mcp__claude-in-chrome__* tools.

### Available skills

- /office-hours
- /plan-ceo-review
- /plan-eng-review
- /plan-design-review
- /design-consultation
- /design-shotgun
- /design-html
- /review
- /ship
- /land-and-deploy
- /canary
- /benchmark
- /browse
- /connect-chrome
- /qa
- /qa-only
- /design-review
- /setup-browser-cookies
- /setup-deploy
- /setup-gbrain
- /retro
- /investigate
- /document-release
- /document-generate
- /codex
- /cso
- /autoplan
- /plan-devex-review
- /devex-review
- /careful
- /freeze
- /guard
- /unfreeze
- /gstack-upgrade
- /learn
