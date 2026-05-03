import argparse
from pathlib import Path
import sys

for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists() and (parent / "model.py").exists():
        sys.path.insert(0, str(parent))
        break

import numpy as np
import pandas as pd

from src.data_utils import sequence_split
from src.metrics import binary_metrics
from src.probes import (
    mahalanobis_fit,
    mahalanobis_score,
    save_pickle,
    standardize_apply,
    standardize_fit,
    train_laplace_from_logistic,
    train_logistic,
    train_ridge,
)


def load_metrics(activation_dir: Path) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in sorted(activation_dir.glob("*/metrics.csv"))]
    if not frames:
        raise FileNotFoundError(f"no metrics.csv files under {activation_dir}")
    return pd.concat(frames, ignore_index=True)


def load_layer(activation_dir: Path, layer: int, metrics: pd.DataFrame) -> np.ndarray:
    chunks = []
    for set_name in metrics["set_name"].drop_duplicates():
        # TODO[DOWSING-04]: Load one set's layer activations and check alignment.
        # The saved .npy rows must match that set's metrics.csv rows exactly;
        # otherwise labels and activations would be paired with the wrong tokens.
        # Cast to float32 before training sklearn probes.
        raise NotImplementedError
    return np.concatenate(chunks, axis=0)


def labels_for_task(metrics: pd.DataFrame, task: str) -> np.ndarray:
    # TODO[DOWSING-04]: Build binary labels for the requested probe task.
    # - ood: positive means the row came from any non-id_clean eval set.
    # - high_loss: positive means loss is at least the 80th percentile of clean
    #   ID loss, marking tokens the model found unusually hard.
    # - confident_failure: positive means high_loss but entropy is no higher than
    #   the clean ID median, a rough "wrong or struggling while confident" signal.
    raise NotImplementedError


def downsample_train(mask: np.ndarray, y: np.ndarray, max_train_tokens: int | None, seed: int = 1337) -> np.ndarray:
    # TODO[DOWSING-04]: Optionally cap the number of training tokens.
    # When max_train_tokens is set, sample from rows where mask is True while
    # trying to keep both positive and negative examples represented. Return a
    # new boolean mask over all rows.
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["ood", "high_loss", "confident_failure"], default="ood")
    parser.add_argument("--activation_dir", default="artifacts/activations")
    parser.add_argument("--out_dir", default="artifacts")
    parser.add_argument("--max_train_tokens", type=int, default=None)
    parser.add_argument("--laplace_samples", type=int, default=64)
    args = parser.parse_args()

    activation_dir = Path(args.activation_dir)
    out_root = Path(args.out_dir)
    metrics = load_metrics(activation_dir)
    y = labels_for_task(metrics, args.task)

    # TODO[DOWSING-04]: Split by sequence, not by individual token.
    # Build keys like "set_name:sequence_id", convert them to integer IDs, and
    # pass those to sequence_split(). This keeps neighboring tokens from the same
    # source window from leaking across train/test.
    raise NotImplementedError

    rows = []
    for layer in range(n_layers):
        x = load_layer(activation_dir, layer, metrics)
        # TODO[DOWSING-04]: Train and evaluate every probe for this layer.
        # Fit standardization on train rows, apply it to all rows, then train
        # logistic, ridge, Laplace-from-logistic, and Mahalanobis probes. Score
        # only test rows and record binary_metrics for deterministic logistic,
        # ridge, Mahalanobis, activation norm, Laplace predictive mean, and
        # Laplace mutual information.
        raise NotImplementedError

        save_pickle(
            {
                "task": args.task,
                "layer": layer,
                "mean": mean,
                "std": std,
                "logistic": clf,
                "ridge": ridge,
                "laplace": laplace,
                "mahalanobis": maha,
            },
            out_root / "probes" / args.task / f"layer_{layer:02d}.pkl",
        )

    result_dir = out_root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(result_dir / f"{args.task}_layerwise.csv", index=False)


if __name__ == "__main__":
    main()
