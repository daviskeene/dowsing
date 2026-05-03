from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score, roc_curve


def binary_metrics(y_true, score, prob=None) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    calibrated_prob = None if prob is None else np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    out = {
        "auroc": safe_auc(y_true, score),
        "auprc": safe_auprc(y_true, score),
        "brier": float("nan") if calibrated_prob is None else safe_brier(y_true, calibrated_prob),
        "nll": float("nan") if calibrated_prob is None else safe_nll(y_true, calibrated_prob),
        "ece": float("nan") if calibrated_prob is None else expected_calibration_error(y_true, calibrated_prob),
        "recall_at_5fpr": recall_at_fpr(y_true, score, 0.05),
    }
    return out


def safe_auc(y_true, score) -> float:
    if len(np.unique(y_true)) < 2:
        # Some small experiment slices contain only one class; undefined metrics stay explicit as NaN.
        return float("nan")
    return float(roc_auc_score(y_true, score))


def safe_auprc(y_true, score) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, score))


def safe_brier(y_true, prob) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(brier_score_loss(y_true, prob))


def safe_nll(y_true, prob) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(log_loss(y_true, prob, labels=[0, 1]))


def expected_calibration_error(y_true, prob, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (prob >= lo) & (prob < hi if hi < 1.0 else prob <= hi)
        if not np.any(mask):
            continue
        ece += mask.mean() * abs(prob[mask].mean() - y_true[mask].mean())
    return float(ece)


def recall_at_fpr(y_true, score, target_fpr: float = 0.05) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, score)
    valid = tpr[fpr <= target_fpr]
    return float(valid.max()) if len(valid) else 0.0
