import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data_utils import sequence_split
from src.metrics import binary_metrics
from scripts.train_probes import labels_for_task, load_metrics


def oriented_baselines(task: str, metrics: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray | None]]:
    entropy = metrics["entropy"].to_numpy()
    max_prob = metrics["max_prob"].to_numpy()
    target_prob = metrics["target_prob"].to_numpy()
    loss = metrics["loss"].to_numpy()
    # Scores are oriented so larger values mean "more likely positive" for AUROC.
    baselines = [
        ("entropy", entropy, None),
        ("neg_max_prob", -max_prob, None),
        ("neg_target_prob", -target_prob, None),
    ]
    if task == "ood":
        baselines.append(("loss", loss, None))
    return baselines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["ood", "high_loss", "confident_failure"], default="ood")
    parser.add_argument("--activation_dir", default="artifacts/activations")
    parser.add_argument("--results_dir", default="artifacts/results")
    args = parser.parse_args()

    metrics = load_metrics(Path(args.activation_dir))
    y = labels_for_task(metrics, args.task)
    seq_keys = (metrics["set_name"].astype(str) + ":" + metrics["sequence_id"].astype(str)).astype("category").cat.codes.to_numpy()
    test_mask = sequence_split(seq_keys)["test"]

    rows = []
    for name, score, prob in oriented_baselines(args.task, metrics):
        row = {"task": args.task, "method": name, "n_test": int(test_mask.sum())}
        row.update(binary_metrics(y[test_mask], score[test_mask], prob=prob[test_mask] if prob is not None else None))
        rows.append(row)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "baselines.csv"
    frame = pd.DataFrame(rows)
    if out_path.exists():
        old = pd.read_csv(out_path)
        old = old[old["task"] != args.task]
        frame = pd.concat([old, frame], ignore_index=True)
    frame.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
