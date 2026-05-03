import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from scripts.train_probes import load_layer, load_metrics
from src.data_utils import sequence_split
from src.metrics import binary_metrics
from src.probes import (
    mahalanobis_fit,
    mahalanobis_score,
    standardize_apply,
    standardize_fit,
    train_laplace_from_logistic,
    train_logistic,
    train_ridge,
)


def split_masks(metrics: pd.DataFrame) -> dict[str, np.ndarray]:
    seq_keys = (
        metrics["set_name"].astype(str) + ":" + metrics["sequence_id"].astype(str)
    ).astype("category").cat.codes.to_numpy()
    return sequence_split(seq_keys)


def n_layers(activation_dir: Path, metrics: pd.DataFrame) -> int:
    set_name = metrics["set_name"].iloc[0]
    return len(list((activation_dir / set_name).glob("layer_*.npy")))


def add_metric_row(rows: list[dict], base: dict, y_true, score, prob=None) -> None:
    row = dict(base)
    row.update(binary_metrics(y_true, score, prob=prob))
    rows.append(row)


def run_leave_one_ood_type_out(
    activation_dir: Path,
    results_dir: Path,
    metrics: pd.DataFrame,
    splits: dict[str, np.ndarray],
    laplace_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ood_types = sorted(t for t in metrics["ood_type"].unique() if t != "id_clean")
    set_ood_type = metrics["ood_type"].to_numpy()
    is_id = set_ood_type == "id_clean"
    is_ood = metrics["is_ood"].to_numpy(dtype=int)
    rows = []

    for holdout in ood_types:
        heldout = set_ood_type == holdout
        # Train on ID plus known OOD families, then test on ID plus the unseen
        # held-out family to measure shift generalization rather than memorization.
        train_mask = splits["train"] & (is_id | ((set_ood_type != "id_clean") & (set_ood_type != holdout)))
        test_mask = splits["test"] & (is_id | heldout)
        y_test = is_ood[test_mask]

        baseline_scores = {
            "entropy": metrics["entropy"].to_numpy(),
            "neg_max_prob": -metrics["max_prob"].to_numpy(),
            "loss": metrics["loss"].to_numpy(),
        }
        for method, score in baseline_scores.items():
            add_metric_row(
                rows,
                {
                    "eval": "leave_one_ood_type_out",
                    "holdout_ood_type": holdout,
                    "method": method,
                    "layer": np.nan,
                    "n_train": int(train_mask.sum()),
                    "n_test": int(test_mask.sum()),
                },
                y_test,
                score[test_mask],
            )

        for layer in range(n_layers(activation_dir, metrics)):
            x = load_layer(activation_dir, layer, metrics)
            mean, std = standardize_fit(x[train_mask])
            xz = standardize_apply(x, mean, std)
            y_train = is_ood[train_mask]

            clf = train_logistic(xz[train_mask], y_train)
            ridge = train_ridge(xz[train_mask], y_train)
            laplace = train_laplace_from_logistic(clf, xz[train_mask])
            maha = mahalanobis_fit(xz[train_mask & (is_ood == 0)])

            det_score = clf.predict_proba(xz[test_mask])[:, 1]
            ridge_score = ridge.decision_function(xz[test_mask])
            laplace_score = laplace.predict(xz[test_mask], num_samples=laplace_samples)["p_mean"]
            maha_score = mahalanobis_score(xz[test_mask], maha)
            norm_score = np.linalg.norm(xz[test_mask], axis=1)

            for method, score, prob in [
                ("deterministic_logistic", det_score, det_score),
                ("laplace_predictive_mean", laplace_score, laplace_score),
                ("ridge_classifier", ridge_score, None),
                ("mahalanobis", maha_score, None),
                ("standardized_activation_norm", norm_score, None),
            ]:
                add_metric_row(
                    rows,
                    {
                        "eval": "leave_one_ood_type_out",
                        "holdout_ood_type": holdout,
                        "method": method,
                        "layer": layer,
                        "n_train": int(train_mask.sum()),
                        "n_test": int(test_mask.sum()),
                    },
                    y_test,
                    score,
                    prob=prob,
                )

    layerwise = pd.DataFrame(rows)
    layerwise.to_csv(results_dir / "ood_leave_one_type_out_layerwise.csv", index=False)

    summary_rows = []
    for holdout in ood_types:
        sub = layerwise[layerwise["holdout_ood_type"] == holdout]
        for method in sub["method"].unique():
            method_rows = sub[sub["method"] == method]
            if method_rows["layer"].notna().any():
                best = method_rows.sort_values("auroc", ascending=False).iloc[0]
            else:
                best = method_rows.iloc[0]
            summary_rows.append(best.to_dict())
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(results_dir / "ood_leave_one_type_out_summary.csv", index=False)
    return layerwise, summary


def run_id_clean_high_loss(
    activation_dir: Path,
    results_dir: Path,
    metrics: pd.DataFrame,
    splits: dict[str, np.ndarray],
    laplace_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    is_id = metrics["set_name"].eq("id_clean").to_numpy()
    train_mask = splits["train"] & is_id
    test_mask = splits["test"] & is_id
    # Fit the surprise threshold on clean train tokens only; this keeps the task
    # about within-distribution high loss, not OOD detection.
    threshold = float(np.quantile(metrics.loc[train_mask, "loss"].to_numpy(), 0.80))
    y = (metrics["loss"].to_numpy() >= threshold).astype(int)
    y_test = y[test_mask]
    rows = []

    for method, score in {
        "entropy": metrics["entropy"].to_numpy(),
        "neg_max_prob": -metrics["max_prob"].to_numpy(),
    }.items():
        add_metric_row(
            rows,
            {
                "eval": "id_clean_high_loss",
                "method": method,
                "layer": np.nan,
                "threshold_source": "id_clean_train_q80",
                "loss_threshold": threshold,
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
            },
            y_test,
            score[test_mask],
        )

    for layer in range(n_layers(activation_dir, metrics)):
        x = load_layer(activation_dir, layer, metrics)
        mean, std = standardize_fit(x[train_mask])
        xz = standardize_apply(x, mean, std)

        clf = train_logistic(xz[train_mask], y[train_mask])
        ridge = train_ridge(xz[train_mask], y[train_mask])
        laplace = train_laplace_from_logistic(clf, xz[train_mask])
        maha = mahalanobis_fit(xz[train_mask & (y == 0)])

        det_score = clf.predict_proba(xz[test_mask])[:, 1]
        ridge_score = ridge.decision_function(xz[test_mask])
        laplace_score = laplace.predict(xz[test_mask], num_samples=laplace_samples)["p_mean"]
        maha_score = mahalanobis_score(xz[test_mask], maha)
        norm_score = np.linalg.norm(xz[test_mask], axis=1)

        for method, score, prob in [
            ("deterministic_logistic", det_score, det_score),
            ("laplace_predictive_mean", laplace_score, laplace_score),
            ("ridge_classifier", ridge_score, None),
            ("mahalanobis", maha_score, None),
            ("standardized_activation_norm", norm_score, None),
        ]:
            add_metric_row(
                rows,
                {
                    "eval": "id_clean_high_loss",
                    "method": method,
                    "layer": layer,
                    "threshold_source": "id_clean_train_q80",
                    "loss_threshold": threshold,
                    "n_train": int(train_mask.sum()),
                    "n_test": int(test_mask.sum()),
                },
                y_test,
                score,
                prob=prob,
            )

    layerwise = pd.DataFrame(rows)
    layerwise.to_csv(results_dir / "high_loss_id_clean_only_layerwise.csv", index=False)
    summary_rows = []
    for method in layerwise["method"].unique():
        method_rows = layerwise[layerwise["method"] == method]
        if method_rows["layer"].notna().any():
            best = method_rows.sort_values("auroc", ascending=False).iloc[0]
        else:
            best = method_rows.iloc[0]
        summary_rows.append(best.to_dict())
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(results_dir / "high_loss_id_clean_only_summary.csv", index=False)
    return layerwise, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation_dir", default="artifacts/activations")
    parser.add_argument("--results_dir", default="artifacts/results")
    parser.add_argument("--laplace_samples", type=int, default=64)
    args = parser.parse_args()

    activation_dir = Path(args.activation_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics(activation_dir)
    splits = split_masks(metrics)
    run_leave_one_ood_type_out(activation_dir, results_dir, metrics, splits, args.laplace_samples)
    run_id_clean_high_loss(activation_dir, results_dir, metrics, splits, args.laplace_samples)
    print(f"wrote robustness evaluations to {results_dir}")


if __name__ == "__main__":
    main()
