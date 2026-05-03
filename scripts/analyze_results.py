import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scripts.train_probes import labels_for_task, load_layer, load_metrics
from src.data_utils import decode, read_tokens, sequence_split
from src.metrics import binary_metrics
from src.plotting import savefig
from src.probes import load_pickle, mahalanobis_score, standardize_apply


DISPLAY_NAMES = {
    "entropy": "Entropy",
    "neg_max_prob": "1 - max prob",
    "loss": "Loss",
    "deterministic_logistic": "Layer probe",
    "laplace_predictive_mean": "Laplace mean",
    "laplace_mutual_information": "Laplace MI",
    "mahalanobis": "Mahalanobis",
}


def markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    if df.empty:
        return "_Not generated._"
    shown = df[cols].copy()
    for col in shown.select_dtypes(include="number").columns:
        if col in {"layer", "n", "n_test"}:
            shown[col] = shown[col].map(lambda x: str(int(x)) if pd.notna(x) else "")
        else:
            shown[col] = shown[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in shown.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def split_masks(metrics: pd.DataFrame) -> dict[str, np.ndarray]:
    seq_keys = (
        metrics["set_name"].astype(str) + ":" + metrics["sequence_id"].astype(str)
    ).astype("category").cat.codes.to_numpy()
    return sequence_split(seq_keys)


def best_layer(results_dir: Path, task: str, probe_type: str = "deterministic_logistic") -> int:
    df = pd.read_csv(results_dir / f"{task}_layerwise.csv")
    sub = df[df["probe_type"] == probe_type].sort_values("auroc", ascending=False)
    return int(sub.iloc[0]["layer"])


def probe_scores(
    activation_dir: Path,
    metrics: pd.DataFrame,
    task: str,
    layer: int,
    probe_type: str,
    seed: int = 1337,
) -> np.ndarray:
    # Probe artifacts store the layer standardization stats with each classifier,
    # so analysis can replay scores without retraining.
    probe = load_pickle(Path("artifacts/probes") / task / f"layer_{layer:02d}.pkl")
    x = load_layer(activation_dir, layer, metrics)
    xz = standardize_apply(x, probe["mean"], probe["std"])
    if probe_type == "deterministic_logistic":
        return probe["logistic"].predict_proba(xz)[:, 1]
    if probe_type == "laplace_predictive_mean":
        return probe["laplace"].predict(xz, seed=seed)["p_mean"]
    if probe_type == "laplace_mutual_information":
        return probe["laplace"].predict(xz, seed=seed)["mutual_information"]
    if probe_type == "mahalanobis":
        return mahalanobis_score(xz, probe["mahalanobis"])
    raise ValueError(f"unknown probe type: {probe_type}")


def make_ood_type_breakdown(
    activation_dir: Path,
    results_dir: Path,
    plots_dir: Path,
    metrics: pd.DataFrame,
    test_mask: np.ndarray,
) -> pd.DataFrame:
    layer = best_layer(results_dir, "ood")
    scores = {
        "entropy": metrics["entropy"].to_numpy(),
        "neg_max_prob": -metrics["max_prob"].to_numpy(),
        "loss": metrics["loss"].to_numpy(),
        "deterministic_logistic": probe_scores(activation_dir, metrics, "ood", layer, "deterministic_logistic"),
        "laplace_predictive_mean": probe_scores(activation_dir, metrics, "ood", layer, "laplace_predictive_mean"),
    }

    rows = []
    id_mask = metrics["ood_type"].eq("id_clean").to_numpy()
    # Each row compares one OOD family against clean ID tokens on the held-out split.
    for ood_type in sorted(t for t in metrics["ood_type"].unique() if t != "id_clean"):
        type_mask = metrics["ood_type"].eq(ood_type).to_numpy()
        mask = test_mask & (id_mask | type_mask)
        y = metrics.loc[mask, "is_ood"].to_numpy()
        for method, score in scores.items():
            row = {
                "ood_type": ood_type,
                "method": method,
                "display_name": DISPLAY_NAMES.get(method, method),
                "best_layer": layer if method in {"deterministic_logistic", "laplace_predictive_mean"} else np.nan,
                "n_test": int(mask.sum()),
            }
            row.update(binary_metrics(y, score[mask]))
            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(results_dir / "ood_type_breakdown.csv", index=False)

    plot_methods = ["entropy", "loss", "deterministic_logistic", "laplace_predictive_mean"]
    pivot = out[out["method"].isin(plot_methods)].pivot(index="ood_type", columns="display_name", values="auroc")
    pivot = pivot[[c for c in ["Entropy", "Loss", "Layer probe", "Laplace mean"] if c in pivot.columns]]
    pivot.plot(kind="bar", figsize=(9, 4.8), width=0.78)
    plt.ylabel("AUROC")
    plt.xlabel("OOD type")
    plt.ylim(0.0, 1.02)
    plt.title("OOD Detection By Shift Type")
    plt.xticks(rotation=30, ha="right")
    savefig(plots_dir / "ood_type_breakdown.png")

    det = out[out["method"] == "deterministic_logistic"][["ood_type", "auroc"]].rename(columns={"auroc": "probe_auroc"})
    entropy = out[out["method"] == "entropy"][["ood_type", "auroc"]].rename(columns={"auroc": "entropy_auroc"})
    delta = det.merge(entropy, on="ood_type")
    delta["delta"] = delta["probe_auroc"] - delta["entropy_auroc"]
    delta = delta.sort_values("delta")
    colors = ["#9b2c2c" if x < 0 else "#1f6f5b" for x in delta["delta"]]
    plt.figure(figsize=(7, 3.8))
    plt.barh(delta["ood_type"], delta["delta"], color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("AUROC improvement over entropy")
    plt.title("Layer Probe Gain By OOD Type")
    savefig(plots_dir / "ood_probe_delta_by_type.png")
    return out


def make_layer_type_heatmap(
    activation_dir: Path,
    results_dir: Path,
    plots_dir: Path,
    metrics: pd.DataFrame,
    test_mask: np.ndarray,
) -> pd.DataFrame:
    n_layers = len(list((activation_dir / metrics["set_name"].iloc[0]).glob("layer_*.npy")))
    id_mask = metrics["ood_type"].eq("id_clean").to_numpy()
    rows = []
    for layer in range(n_layers):
        score = probe_scores(activation_dir, metrics, "ood", layer, "deterministic_logistic")
        for ood_type in sorted(t for t in metrics["ood_type"].unique() if t != "id_clean"):
            type_mask = metrics["ood_type"].eq(ood_type).to_numpy()
            mask = test_mask & (id_mask | type_mask)
            y = metrics.loc[mask, "is_ood"].to_numpy()
            row = {"layer": layer, "ood_type": ood_type, "n_test": int(mask.sum())}
            row.update(binary_metrics(y, score[mask]))
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(results_dir / "ood_layer_by_type.csv", index=False)

    pivot = out.pivot(index="layer", columns="ood_type", values="auroc")
    plt.figure(figsize=(8, 4.5))
    plt.imshow(pivot.to_numpy(), aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.colorbar(label="AUROC")
    plt.xlabel("OOD type")
    plt.ylabel("Layer")
    plt.title("Layerwise OOD Probe AUROC By Shift Type")
    savefig(plots_dir / "ood_layer_type_heatmap.png")
    return out


def make_fair_baselines(results_dir: Path) -> pd.DataFrame:
    rows = []
    baselines = pd.read_csv(results_dir / "baselines.csv")
    for task in ("ood", "high_loss"):
        layerwise = pd.read_csv(results_dir / f"{task}_layerwise.csv")
        for _, row in baselines[baselines["task"] == task].iterrows():
            if task == "high_loss" and row["method"] == "neg_target_prob":
                continue
            rows.append(
                {
                    "task": task,
                    "method": row["method"],
                    "layer": np.nan,
                    "auroc": row["auroc"],
                    "auprc": row["auprc"],
                    "brier": row["brier"],
                    "ece": row["ece"],
                    "note": "output baseline",
                }
            )
        for probe_type in [
            "activation_norm",
            "mahalanobis",
            "deterministic_logistic",
            "laplace_predictive_mean",
            "laplace_mutual_information",
        ]:
            best = layerwise[layerwise["probe_type"] == probe_type].sort_values("auroc", ascending=False).iloc[0]
            rows.append(
                {
                    "task": task,
                    "method": probe_type,
                    "layer": int(best["layer"]),
                    "auroc": best["auroc"],
                    "auprc": best["auprc"],
                    "brier": best["brier"],
                    "ece": best["ece"],
                    "note": "best layer",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(results_dir / "fair_baselines.csv", index=False)
    return out


def read_context(dataset_dir: Path, row: pd.Series, left: int = 96, right: int = 16) -> str:
    split_path = dataset_dir / "eval" / f"{row['set_name']}.bin"
    tokens = np.asarray(read_tokens(split_path), dtype=np.uint16)
    target_pos = int(row["source_start"]) + int(row["position"]) + 1
    start = max(0, target_pos - left)
    end = min(len(tokens), target_pos + right)
    text = decode(tokens[start:end])
    target_offset = target_pos - start
    if 0 <= target_offset < len(text):
        text = text[:target_offset] + "[[" + text[target_offset : target_offset + 1] + "]]" + text[target_offset + 1 :]
    return text.replace("\n", "\\n")


def make_qualitative_examples(
    activation_dir: Path,
    dataset_dir: Path,
    results_dir: Path,
    plots_dir: Path,
    metrics: pd.DataFrame,
    test_mask: np.ndarray,
) -> pd.DataFrame:
    layer = best_layer(results_dir, "high_loss")
    score = probe_scores(activation_dir, metrics, "high_loss", layer, "deterministic_logistic")
    id_metrics = metrics[metrics["set_name"] == "id_clean"]
    loss_threshold = id_metrics["loss"].quantile(0.80)
    entropy_median = id_metrics["entropy"].median()
    # Confident failures are high-loss tokens where the model's output distribution
    # is not unusually uncertain, so entropy alone is a weak warning signal.
    confident_failure = (
        test_mask
        & (metrics["loss"].to_numpy() >= loss_threshold)
        & (metrics["entropy"].to_numpy() <= entropy_median)
    )
    score_series = pd.Series(score)
    catch_cutoff = np.quantile(score[confident_failure], 0.80) if confident_failure.any() else np.quantile(score, 0.80)
    miss_cutoff = np.quantile(score[confident_failure], 0.20) if confident_failure.any() else np.quantile(score, 0.20)
    work = metrics.copy()
    work["probe_score"] = score
    work["confident_failure"] = confident_failure
    catches = work[confident_failure & (score_series >= catch_cutoff)].sort_values(["loss", "probe_score"], ascending=False).head(5)
    misses = work[confident_failure & (score_series <= miss_cutoff)].sort_values(["loss", "probe_score"], ascending=[False, True]).head(5)
    examples = []
    for label, frame in [("probe_catch", catches), ("probe_miss", misses)]:
        for _, row in frame.iterrows():
            target_id = int(row["target_id"])
            top_pred_id = int(row["top_pred_id"]) if "top_pred_id" in row and pd.notna(row["top_pred_id"]) else -1
            examples.append(
                {
                    "case": label,
                    "set_name": row["set_name"],
                    "context": read_context(dataset_dir, row),
                    "target": repr(chr(target_id)) if 0 <= target_id < 128 else str(target_id),
                    "top_prediction": repr(chr(top_pred_id)) if 0 <= top_pred_id < 128 else str(top_pred_id),
                    "loss": row["loss"],
                    "entropy": row["entropy"],
                    "max_prob": row["max_prob"],
                    "probe_score": row["probe_score"],
                }
            )
    out = pd.DataFrame(examples)
    out.to_csv(results_dir / "qualitative_examples.csv", index=False)
    md = "## Qualitative Confident-Failure Examples\n\n"
    md += markdown_table(out, ["case", "set_name", "target", "top_prediction", "loss", "entropy", "probe_score"])
    md += "\n"
    (results_dir / "qualitative_examples.md").write_text(md, encoding="utf-8")

    sample = work[test_mask].sample(n=min(8000, int(test_mask.sum())), random_state=1337)
    plt.figure(figsize=(6.5, 5.2))
    plt.scatter(sample["entropy"], sample["loss"], c=sample["probe_score"], s=6, cmap="viridis", alpha=0.65)
    plt.axhline(loss_threshold, color="#9b2c2c", linewidth=1.0)
    plt.axvline(entropy_median, color="#9b2c2c", linewidth=1.0)
    plt.xlabel("Output entropy")
    plt.ylabel("Next-token loss")
    plt.title(f"Confident Failures Colored By Layer {layer} Probe")
    plt.colorbar(label="Probe high-loss score")
    savefig(plots_dir / "confident_failures_probe_score.png")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation_dir", default="artifacts/activations")
    parser.add_argument("--dataset_dir", default="data/uncertainty_byte")
    parser.add_argument("--results_dir", default="artifacts/results")
    parser.add_argument("--plots_dir", default="artifacts/plots")
    args = parser.parse_args()

    activation_dir = Path(args.activation_dir)
    dataset_dir = Path(args.dataset_dir)
    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(activation_dir)
    test_mask = split_masks(metrics)["test"]
    make_ood_type_breakdown(activation_dir, results_dir, plots_dir, metrics, test_mask)
    make_layer_type_heatmap(activation_dir, results_dir, plots_dir, metrics, test_mask)
    make_fair_baselines(results_dir)
    make_qualitative_examples(activation_dir, dataset_dir, results_dir, plots_dir, metrics, test_mask)
    print(f"wrote presentation analysis to {results_dir} and {plots_dir}")


if __name__ == "__main__":
    main()
