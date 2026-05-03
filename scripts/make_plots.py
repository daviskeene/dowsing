import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.plotting import savefig


def plot_layerwise(results_dir: Path, plots_dir: Path, task: str) -> None:
    path = results_dir / f"{task}_layerwise.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    plt.figure(figsize=(7, 4))
    styles = [
        ("deterministic_logistic", "Deterministic probe", {"linewidth": 2.6, "marker": "o", "markersize": 7}),
        ("laplace_predictive_mean", "Bayesian mean", {"linewidth": 1.6, "linestyle": "--", "marker": "x", "markersize": 7}),
        ("laplace_mutual_information", "Bayesian MI", {"linewidth": 1.6, "marker": "o", "markersize": 5}),
    ]
    for probe_type, label, kwargs in styles:
        sub = df[df["probe_type"] == probe_type].sort_values("layer")
        if len(sub):
            plt.plot(sub["layer"], sub["auroc"], label=label, **kwargs)

    baselines_path = results_dir / "baselines.csv"
    if baselines_path.exists():
        baselines = pd.read_csv(baselines_path)
        baselines = baselines[baselines["task"] == task]
        for method, label, color in [
            ("entropy", "Output entropy", "#888888"),
            ("neg_max_prob", "1 − max prob", "#bbbbbb"),
        ]:
            row = baselines[baselines["method"] == method]
            if not row.empty:
                plt.axhline(float(row.iloc[0]["auroc"]), color=color, linestyle=":", linewidth=1.5, label=label)

    plt.xlabel("Layer")
    plt.ylabel("AUROC")
    pretty_task = "OOD" if task == "ood" else task.replace("_", " ").title()
    plt.title(f"{pretty_task} AUROC by layer")
    plt.ylim(0.0, 1.02)
    plt.legend(loc="lower right", fontsize=9)
    savefig(plots_dir / f"{task}_auroc_by_layer.png")


def plot_ood_type_breakdown(activation_dir: Path, results_dir: Path, plots_dir: Path) -> None:
    metrics_paths = sorted(activation_dir.glob("*/metrics.csv"))
    if not metrics_paths:
        return
    metrics = pd.concat([pd.read_csv(p) for p in metrics_paths], ignore_index=True)
    layerwise_path = results_dir / "ood_layerwise.csv"
    if not layerwise_path.exists():
        return
    layerwise = pd.read_csv(layerwise_path)
    best_det = layerwise[layerwise["probe_type"] == "deterministic_logistic"].sort_values("auroc", ascending=False).head(1)
    best_bayes = layerwise[layerwise["probe_type"] == "laplace_predictive_mean"].sort_values("auroc", ascending=False).head(1)
    rows = []
    from sklearn.metrics import roc_auc_score
    from src.probes import load_pickle, standardize_apply

    for ood_type in sorted(t for t in metrics["ood_type"].unique() if t != "id_clean"):
        mask = metrics["ood_type"].isin(["id_clean", ood_type]).to_numpy()
        y = metrics.loc[mask, "is_ood"].to_numpy()
        try:
            entropy_auc = roc_auc_score(y, metrics.loc[mask, "entropy"])
        except ValueError:
            entropy_auc = np.nan
        rows.append({"ood_type": ood_type, "method": "entropy", "auroc": entropy_auc})
        for label, best in [("best deterministic probe", best_det), ("best Bayesian probe", best_bayes)]:
            if best.empty:
                continue
            layer = int(best.iloc[0]["layer"])
            probe = load_pickle(Path("artifacts/probes/ood") / f"layer_{layer:02d}.pkl")
            chunks = []
            for set_name in metrics.loc[mask, "set_name"].drop_duplicates():
                chunks.append(np.load(activation_dir / set_name / f"layer_{layer:02d}.npy").astype(np.float32))
            x_all = np.concatenate(chunks, axis=0)
            xz = standardize_apply(x_all, probe["mean"], probe["std"])
            if "Bayesian" in label:
                score = probe["laplace"].predict(xz)["p_mean"]
            else:
                score = probe["logistic"].predict_proba(xz)[:, 1]
            try:
                auc = roc_auc_score(y, score)
            except ValueError:
                auc = np.nan
            rows.append({"ood_type": ood_type, "method": label, "auroc": auc})
    df = pd.DataFrame(rows)
    if df.empty:
        return
    pivot = df.pivot(index="ood_type", columns="method", values="auroc")
    pivot.plot(kind="bar", figsize=(8, 4))
    plt.ylabel("AUROC")
    plt.xlabel("OOD type")
    plt.ylim(0.0, 1.02)
    plt.title("OOD type breakdown")
    savefig(plots_dir / "ood_type_breakdown.png")


def plot_confident_failures(activation_dir: Path, plots_dir: Path) -> None:
    paths = sorted(activation_dir.glob("*/metrics.csv"))
    if not paths:
        return
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    score = df["loss"].rank(pct=True).to_numpy()
    plt.figure(figsize=(6, 5))
    sample = df.sample(n=min(len(df), 5000), random_state=1337)
    plt.scatter(sample["entropy"], sample["loss"], c=score[sample.index], s=6, cmap="viridis", alpha=0.65)
    id_df = df[df["set_name"] == "id_clean"]
    plt.axhline(id_df["loss"].quantile(0.80), color="red", linewidth=1)
    plt.axvline(id_df["entropy"].median(), color="red", linewidth=1)
    plt.xlabel("Output entropy")
    plt.ylabel("Next-token loss")
    plt.title("Confident failure region")
    plt.colorbar(label="Loss percentile")
    savefig(plots_dir / "confident_failures.png")


def plot_heatmap(results_dir: Path, plots_dir: Path) -> None:
    path = results_dir / "ood_layerwise.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    sub = df[df["probe_type"].isin(["deterministic_logistic", "laplace_predictive_mean"])]
    pivot = sub.pivot(index="layer", columns="probe_type", values="auroc")
    plt.figure(figsize=(5, 4))
    plt.imshow(pivot.to_numpy(), aspect="auto", vmin=0.0, vmax=1.0, cmap="magma")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xticks(range(len(pivot.columns)), [c.replace("_", "\n") for c in pivot.columns], fontsize=8)
    plt.colorbar(label="AUROC")
    plt.title("Layer/probe AUROC heatmap")
    savefig(plots_dir / "layer_heatmap.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation_dir", default="artifacts/activations")
    parser.add_argument("--results_dir", default="artifacts/results")
    parser.add_argument("--plots_dir", default="artifacts/plots")
    args = parser.parse_args()
    activation_dir = Path(args.activation_dir)
    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir)
    plot_layerwise(results_dir, plots_dir, "ood")
    plot_layerwise(results_dir, plots_dir, "high_loss")
    plot_ood_type_breakdown(activation_dir, results_dir, plots_dir)
    plot_confident_failures(activation_dir, plots_dir)
    plot_heatmap(results_dir, plots_dir)


if __name__ == "__main__":
    main()
