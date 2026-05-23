# Chest X-ray classifier

A multi-label chest X-ray classifier built with transfer learning on the NIH ChestX-ray14 dataset.

It fine-tunes a DenseNet-121 to predict 14 findings per image, each as an independent probability.

## Not a medical device

This is a learning and portfolio project. It is NOT a clinical or diagnostic tool.

The labels come from text-mined radiology reports, not from doctors confirming the images, so they are noisy. The model is not clinically validated and must never be used to diagnose patients.

## Status

Early development. See PROGRESS.md for what is done and what is left.

## Project layout

- config.yaml: all hyperparameters
- src/config.py: loads the config
- src/data.py: dataset class and the patient-level split
- (coming) preprocessing, model, training, evaluation modules
- tests/: small synthetic-data checks, including the no-leakage guard

## Dataset

NIH ChestX-ray14. About 112,120 frontal PNG images from about 30,805 patients, 14 findings.

Download the images, Data_Entry_2017.csv, and the official split lists (train_val_list.txt, test_list.txt), then place them under data/. The data folder is gitignored and never committed.

## The one rule that matters most

Split by patient, never by image. The same patient must never appear in more than one of train, validation, or test. The code asserts this and will stop if it is ever violated.
