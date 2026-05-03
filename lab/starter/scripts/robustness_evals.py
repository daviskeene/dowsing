import argparse
from pathlib import Path
import sys

for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists() and (parent / "model.py").exists():
        sys.path.insert(0, str(parent))
        break

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
    # TODO[DOWSING-06]: Make train/val/test masks without token leakage.
    # Tokens from the same set_name:sequence_id are neighbors from one source
    # window, so they must stay in the same split. Build those sequence keys and
    # pass their integer IDs to sequence_split().
    raise NotImplementedError


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
    """Train on ID plus four OOD families, test on held-out OOD family plus ID."""
    # TODO[DOWSING-06]: Test whether probes generalize to unseen OOD families.
    # For each OOD type, train on ID plus the other OOD types and test on ID plus
    # the held-out type. Include simple metric baselines such as entropy/loss and
    # the layerwise probes from earlier milestones. Save the full layerwise CSV
    # and a summary CSV with the best layer per method.
    raise NotImplementedError


def run_id_clean_high_loss(
    activation_dir: Path,
    results_dir: Path,
    metrics: pd.DataFrame,
    splits: dict[str, np.ndarray],
    laplace_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train/test high-loss probes only on held-out Shakespeare rows."""
    # TODO[DOWSING-06]: Probe hard clean-text tokens without using OOD labels.
    # Fit the high-loss threshold from ID train rows only, then evaluate on ID
    # test rows. This asks whether activations predict ordinary model failures,
    # not just distribution shift. Save both layerwise and summary CSVs.
    raise NotImplementedError


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
