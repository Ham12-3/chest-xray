"""Evaluation metrics for the multi-label task.

Lead metric is AUC-ROC per finding, plus the macro average (hard rule 3).
We also offer precision and recall at a threshold, and a bootstrap confidence
interval for the macro AUC, because rare findings give noisy AUC.

Accuracy is intentionally left out. Under this much imbalance it is
misleading: a model that predicts all-negative looks "accurate" but is useless.
"""

import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from src.data import LABELS


def per_finding_auc(y_true, y_prob):
    """AUC per finding. nan when a finding has only one class in y_true."""
    aucs = {}
    for i, label in enumerate(LABELS):
        col = y_true[:, i]
        if col.min() == col.max():
            aucs[label] = float("nan")  # AUC is undefined with one class only
        else:
            aucs[label] = float(roc_auc_score(col, y_prob[:, i]))
    return aucs


def macro_auc(aucs):
    """Mean AUC over findings that had a defined AUC."""
    vals = [v for v in aucs.values() if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def bootstrap_macro_auc(y_true, y_prob, n_samples=1000, seed=0):
    """95 percent confidence interval for the macro AUC, by resampling rows."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    samples = []
    for _ in range(n_samples):
        idx = rng.integers(0, n, n)
        samples.append(macro_auc(per_finding_auc(y_true[idx], y_prob[idx])))
    lo, hi = np.nanpercentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def precision_recall_at(y_true, y_prob, thresholds):
    """Per-finding precision and recall at the given thresholds.

    thresholds is a 1D array, one per finding. Choose these on validation,
    never on test.
    """
    out = {}
    for i, label in enumerate(LABELS):
        pred = (y_prob[:, i] >= thresholds[i]).astype(int)
        out[label] = {
            "precision": float(precision_score(y_true[:, i], pred, zero_division=0)),
            "recall": float(recall_score(y_true[:, i], pred, zero_division=0)),
        }
    return out


def format_report(aucs):
    """Pretty per-finding AUC table with the macro average at the bottom."""
    lines = [f"  {'finding':20s} {'AUC':>6s}"]
    for label, auc in aucs.items():
        shown = "  nan" if np.isnan(auc) else f"{auc:.3f}"
        lines.append(f"  {label:20s} {shown:>6s}")
    lines.append(f"  {'MACRO':20s} {macro_auc(aucs):.3f}")
    return "\n".join(lines)
