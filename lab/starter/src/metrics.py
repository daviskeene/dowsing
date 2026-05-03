import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score, roc_curve


def binary_metrics(y_true, score, prob=None) -> dict[str, float]:
    """Return ranking metrics for any score and calibration metrics only for probabilities."""
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)

    # TODO[DOWSING-03]: Collect ranking and calibration metrics in one dict.
    # AUROC/AUPRC/recall@FPR can use any score where larger means "more likely
    # positive." Brier, NLL, and ECE need probabilities, so only compute them
    # when prob is provided. Clip probabilities into [1e-6, 1 - 1e-6] before
    # log-based metrics; if prob is None, return NaN for the calibration fields.
    raise NotImplementedError


def safe_auc(y_true, score) -> float:
    # TODO[DOWSING-03]: Wrap roc_auc_score for small slices.
    # AUROC is undefined when y_true contains only one class; return NaN instead
    # of raising so experiment tables can still be written.
    raise NotImplementedError


def safe_auprc(y_true, score) -> float:
    # TODO[DOWSING-03]: Same guard as safe_auc, but for average precision.
    raise NotImplementedError


def safe_brier(y_true, prob) -> float:
    # TODO[DOWSING-03]: Compute Brier score when both classes are present.
    # The input here should already be clipped probabilities, not raw margins.
    raise NotImplementedError


def safe_nll(y_true, prob) -> float:
    # TODO[DOWSING-03]: Compute binary log loss when both classes are present.
    # Pass labels=[0, 1] so sklearn keeps the class order stable.
    raise NotImplementedError


def expected_calibration_error(y_true, prob, n_bins: int = 10) -> float:
    # TODO[DOWSING-03]: Implement equal-width-bin expected calibration error.
    # For each probability bin, compare mean predicted probability with the
    # observed positive rate, weight by the bin's fraction of examples, and sum.
    # Include prob == 1.0 in the final bin.
    raise NotImplementedError


def recall_at_fpr(y_true, score, target_fpr: float = 0.05) -> float:
    # TODO[DOWSING-03]: Measure recall under a false-positive-rate budget.
    # Use roc_curve, keep points with FPR <= target_fpr, and return the best TPR
    # among them. As above, return NaN if only one class is present.
    raise NotImplementedError
