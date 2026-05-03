import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    frames = []
    for metrics_path in sorted(activation_dir.glob("*/metrics.csv")):
        frames.append(pd.read_csv(metrics_path))
    if not frames:
        raise FileNotFoundError(f"no metrics.csv files under {activation_dir}")
    return pd.concat(frames, ignore_index=True)


def load_layer(activation_dir: Path, layer: int, metrics: pd.DataFrame) -> np.ndarray:
    chunks = []
    for set_name in metrics["set_name"].drop_duplicates():
        set_metrics = metrics[metrics["set_name"] == set_name]
        path = activation_dir / set_name / f"layer_{layer:02d}.npy"
        layer_chunk = np.load(path).astype(np.float32)
        if len(layer_chunk) != len(set_metrics):
            raise ValueError(
                f"activation/metric row mismatch for {set_name} layer {layer}: "
                f"{len(layer_chunk)} activations vs {len(set_metrics)} metric rows"
            )
        chunks.append(layer_chunk)
    activations = np.concatenate(chunks, axis=0)
    if len(activations) != len(metrics):
        raise ValueError(f"activation/metric row mismatch: {len(activations)} activations vs {len(metrics)} metric rows")
    return activations


def labels_for_task(metrics: pd.DataFrame, task: str) -> np.ndarray:
    id_loss = metrics.loc[metrics["set_name"] == "id_clean", "loss"].to_numpy()
    threshold = np.quantile(id_loss, 0.80)
    median_entropy = metrics.loc[metrics["set_name"] == "id_clean", "entropy"].median()
    # High-loss and confident-failure labels are anchored to clean validation
    # tokens, so their thresholds do not move just because OOD sets are present.
    if task == "ood":
        return metrics["is_ood"].to_numpy(dtype=int)
    if task == "high_loss":
        return (metrics["loss"].to_numpy() >= threshold).astype(int)
    if task == "confident_failure":
        return ((metrics["loss"].to_numpy() >= threshold) & (metrics["entropy"].to_numpy() <= median_entropy)).astype(int)
    raise ValueError(f"unknown task: {task}")


def downsample_train(mask: np.ndarray, y: np.ndarray, max_train_tokens: int | None, seed: int = 1337) -> np.ndarray:
    idx = np.where(mask)[0]
    if max_train_tokens is None or len(idx) <= max_train_tokens:
        return mask
    rng = np.random.default_rng(seed)
    # Keep the limited training set roughly class-balanced when possible; otherwise
    # easy majority-class tokens can dominate the linear probes.
    pos = idx[y[idx] == 1]
    neg = idx[y[idx] == 0]
    half = max_train_tokens // 2
    take_pos = rng.choice(pos, size=min(len(pos), half), replace=False) if len(pos) else np.array([], dtype=int)
    take_neg = rng.choice(neg, size=min(len(neg), max_train_tokens - len(take_pos)), replace=False) if len(neg) else np.array([], dtype=int)
    chosen = np.concatenate([take_pos, take_neg])
    if len(chosen) < max_train_tokens:
        rest = np.setdiff1d(idx, chosen, assume_unique=False)
        extra = rng.choice(rest, size=min(len(rest), max_train_tokens - len(chosen)), replace=False)
        chosen = np.concatenate([chosen, extra])
    out = np.zeros_like(mask, dtype=bool)
    out[chosen] = True
    return out


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
    seq_keys = (metrics["set_name"].astype(str) + ":" + metrics["sequence_id"].astype(str)).astype("category").cat.codes.to_numpy()
    # Split by sampled window, not token, to avoid adjacent tokens leaking between
    # train and test.
    splits = sequence_split(seq_keys)
    train_mask = downsample_train(splits["train"], y, args.max_train_tokens)
    test_mask = splits["test"]

    n_layers = len(list((activation_dir / metrics["set_name"].iloc[0]).glob("layer_*.npy")))
    rows = []
    for layer in range(n_layers):
        x = load_layer(activation_dir, layer, metrics)
        mean, std = standardize_fit(x[train_mask])
        xz = standardize_apply(x, mean, std)
        clf = train_logistic(xz[train_mask], y[train_mask])
        ridge = train_ridge(xz[train_mask], y[train_mask])
        laplace = train_laplace_from_logistic(clf, xz[train_mask])
        maha = mahalanobis_fit(xz[train_mask & (y == 0)])

        det_score = clf.predict_proba(xz[test_mask])[:, 1]
        ridge_score = ridge.decision_function(xz[test_mask])
        maha_score = mahalanobis_score(xz[test_mask], maha)
        norm_score = np.linalg.norm(xz[test_mask], axis=1)
        bayes = laplace.predict(xz[test_mask], num_samples=args.laplace_samples)

        for probe_type, score, prob in [
            ("deterministic_logistic", det_score, det_score),
            ("ridge_classifier", ridge_score, None),
            ("mahalanobis", maha_score, None),
            ("activation_norm", norm_score, None),
            ("laplace_predictive_mean", bayes["p_mean"], bayes["p_mean"]),
            ("laplace_mutual_information", bayes["mutual_information"], None),
        ]:
            row = {
                "layer": layer,
                "task": args.task,
                "probe_type": probe_type,
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
            }
            row.update(binary_metrics(y[test_mask], score, prob=prob))
            rows.append(row)

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
