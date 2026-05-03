import argparse
from pathlib import Path

import pandas as pd


def markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    if df.empty:
        return "_Not generated yet._"
    shown = df[cols].copy()
    for col in shown.select_dtypes(include="number").columns:
        if col in {"layer", "n_train", "n_test", "best_layer", "best_probe_layer"}:
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


def best_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df.sort_values("auroc", ascending=False).groupby("probe_type", as_index=False).head(1)


def best_probe(df: pd.DataFrame, probe_type: str = "deterministic_logistic") -> pd.Series:
    return df[df["probe_type"] == probe_type].sort_values("auroc", ascending=False).iloc[0]


def baseline_value(baselines: pd.DataFrame, task: str, method: str) -> float:
    row = baselines[(baselines["task"] == task) & (baselines["method"] == method)]
    return float(row.iloc[0]["auroc"]) if len(row) else float("nan")


def format_delta(value: float) -> str:
    return f"{value:+.3f}" if pd.notna(value) else "n/a"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="artifacts/results")
    parser.add_argument("--report", default="report/mini_report.md")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    ood = best_rows(results_dir / "ood_layerwise.csv")
    high_loss = best_rows(results_dir / "high_loss_layerwise.csv")
    baselines = pd.read_csv(results_dir / "baselines.csv") if (results_dir / "baselines.csv").exists() else pd.DataFrame()
    fair = pd.read_csv(results_dir / "fair_baselines.csv") if (results_dir / "fair_baselines.csv").exists() else pd.DataFrame()
    ood_types = pd.read_csv(results_dir / "ood_type_breakdown.csv") if (results_dir / "ood_type_breakdown.csv").exists() else pd.DataFrame()
    ood_layer_by_type = pd.read_csv(results_dir / "ood_layer_by_type.csv") if (results_dir / "ood_layer_by_type.csv").exists() else pd.DataFrame()
    loso = pd.read_csv(results_dir / "ood_leave_one_type_out_summary.csv") if (results_dir / "ood_leave_one_type_out_summary.csv").exists() else pd.DataFrame()
    id_high_loss = pd.read_csv(results_dir / "high_loss_id_clean_only_summary.csv") if (results_dir / "high_loss_id_clean_only_summary.csv").exists() else pd.DataFrame()
    qualitative = pd.read_csv(results_dir / "qualitative_examples.csv") if (results_dir / "qualitative_examples.csv").exists() else pd.DataFrame()
    ood_layerwise = pd.read_csv(results_dir / "ood_layerwise.csv") if (results_dir / "ood_layerwise.csv").exists() else pd.DataFrame()
    high_layerwise = pd.read_csv(results_dir / "high_loss_layerwise.csv") if (results_dir / "high_loss_layerwise.csv").exists() else pd.DataFrame()

    ood_best = best_probe(ood_layerwise) if not ood_layerwise.empty else pd.Series(dtype=float)
    high_best = best_probe(high_layerwise) if not high_layerwise.empty else pd.Series(dtype=float)
    ood_entropy = baseline_value(baselines, "ood", "entropy")
    high_entropy = baseline_value(baselines, "high_loss", "entropy")
    high_neg_max = baseline_value(baselines, "high_loss", "neg_max_prob")
    ood_loss = baseline_value(baselines, "ood", "loss")

    ood_type_summary = pd.DataFrame()
    best_ood_type_layers = pd.DataFrame()
    if not ood_types.empty:
        probe = ood_types[ood_types["method"] == "deterministic_logistic"][["ood_type", "auroc"]].rename(columns={"auroc": "layer5_probe_auroc"})
        entropy = ood_types[ood_types["method"] == "entropy"][["ood_type", "auroc"]].rename(columns={"auroc": "entropy_auroc"})
        ood_type_summary = probe.merge(entropy, on="ood_type")
        ood_type_summary["delta"] = ood_type_summary["layer5_probe_auroc"] - ood_type_summary["entropy_auroc"]
        ood_type_summary = ood_type_summary.sort_values("delta", ascending=False)
    if not ood_layer_by_type.empty:
        best_ood_type_layers = (
            ood_layer_by_type.sort_values("auroc", ascending=False)
            .groupby("ood_type", as_index=False)
            .head(1)
            .rename(columns={"layer": "best_layer", "auroc": "best_probe_auroc"})
            .sort_values("ood_type")
        )

    loso_summary = pd.DataFrame()
    loso_best_detector = pd.DataFrame()
    loso_probe_beats_entropy = 0
    loso_total = 0
    loso_char_shuffle_delta = float("nan")
    if not loso.empty:
        det = loso[loso["method"] == "deterministic_logistic"][
            ["holdout_ood_type", "layer", "auroc"]
        ].rename(columns={"layer": "best_probe_layer", "auroc": "probe_auroc"})
        entropy = loso[loso["method"] == "entropy"][["holdout_ood_type", "auroc"]].rename(columns={"auroc": "entropy_auroc"})
        loss = loso[loso["method"] == "loss"][["holdout_ood_type", "auroc"]].rename(columns={"auroc": "loss_auroc"})
        loso_summary = det.merge(entropy, on="holdout_ood_type", how="left").merge(loss, on="holdout_ood_type", how="left")
        loso_summary["delta_vs_entropy"] = loso_summary["probe_auroc"] - loso_summary["entropy_auroc"]
        loso_summary = loso_summary.sort_values("delta_vs_entropy", ascending=False)
        loso_total = int(len(loso_summary))
        loso_probe_beats_entropy = int((loso_summary["delta_vs_entropy"] > 0).sum())
        char_row = loso_summary[loso_summary["holdout_ood_type"] == "char_shuffle"]
        if len(char_row):
            loso_char_shuffle_delta = float(char_row.iloc[0]["delta_vs_entropy"])

        # Exclude oracle baselines from deployable-detector rankings because they
        # need the true next token.
        oracle_methods = {"loss", "neg_target_prob"}
        loso_deployable = loso[~loso["method"].isin(oracle_methods)]
        loso_best_detector = (
            loso_deployable.sort_values("auroc", ascending=False)
            .groupby("holdout_ood_type", as_index=False)
            .head(1)
            .rename(columns={"method": "best_method", "layer": "best_layer", "auroc": "best_auroc"})
            [["holdout_ood_type", "best_method", "best_layer", "best_auroc"]]
            .sort_values("holdout_ood_type")
        )

    id_high_loss_summary = pd.DataFrame()
    id_high_loss_probe_auroc = float("nan")
    id_high_loss_entropy_auroc = float("nan")
    id_high_loss_probe_layer = -1
    if not id_high_loss.empty:
        id_high_loss_summary = id_high_loss[
            id_high_loss["method"].isin(["entropy", "neg_max_prob", "deterministic_logistic", "laplace_predictive_mean", "mahalanobis", "standardized_activation_norm"])
        ][["method", "layer", "auroc", "auprc", "brier", "ece", "n_test"]].copy()
        id_high_loss_summary = id_high_loss_summary.sort_values("auroc", ascending=False)
        det_row = id_high_loss_summary[id_high_loss_summary["method"] == "deterministic_logistic"]
        ent_row = id_high_loss_summary[id_high_loss_summary["method"] == "entropy"]
        if len(det_row):
            id_high_loss_probe_auroc = float(det_row.iloc[0]["auroc"])
            if pd.notna(det_row.iloc[0]["layer"]):
                id_high_loss_probe_layer = int(det_row.iloc[0]["layer"])
        if len(ent_row):
            id_high_loss_entropy_auroc = float(ent_row.iloc[0]["auroc"])

    h1_verdict = "Supported"
    if pd.notna(id_high_loss_probe_auroc) and pd.notna(id_high_loss_entropy_auroc):
        h1_evidence = (
            f"ID-only high-loss: probe layer {id_high_loss_probe_layer} {id_high_loss_probe_auroc:.3f} AUROC "
            f"vs entropy {id_high_loss_entropy_auroc:.3f}"
        )
    else:
        h1_verdict = "Supported in original task; robustness pending"
        h1_evidence = (
            f"original mixed ID/OOD task: best probe {high_best.get('auroc', float('nan')):.3f} AUROC "
            f"vs entropy {high_entropy:.3f}"
        )

    if loso_total:
        h2_verdict = (
            f"Mixed under LOSO: probe beats entropy on {loso_probe_beats_entropy}/{loso_total} held-out shifts"
        )
        h2_evidence_parts = [
            f"known-shift aggregate: probe {ood_best.get('auroc', float('nan')):.3f} vs entropy {ood_entropy:.3f}",
        ]
        if pd.notna(loso_char_shuffle_delta):
            h2_evidence_parts.append(
                f"LOSO char_shuffle delta {format_delta(loso_char_shuffle_delta)} (probe underperforms entropy)"
            )
        if not loso_best_detector.empty:
            non_logistic = loso_best_detector[loso_best_detector["best_method"] != "deterministic_logistic"]
            if len(non_logistic):
                wins = ", ".join(f"{r['holdout_ood_type']}:{r['best_method']}" for _, r in non_logistic.iterrows())
                h2_evidence_parts.append(f"non-logistic best on {wins}")
        h2_evidence = "; ".join(h2_evidence_parts)
    else:
        h2_verdict = "Supported for known shifts; LOSO check pending"
        h2_evidence = f"known-shift aggregate: probe {ood_best.get('auroc', float('nan')):.3f} vs entropy {ood_entropy:.3f}"

    hypothesis_rows = pd.DataFrame(
        [
            {
                "hypothesis": "H1: probes predict high-loss better than entropy",
                "verdict": h1_verdict,
                "evidence": h1_evidence,
            },
            {
                "hypothesis": "H2: probes detect OOD better than entropy",
                "verdict": h2_verdict,
                "evidence": h2_evidence,
            },
            {
                "hypothesis": "H3: probe performance varies by layer",
                "verdict": "Supported",
                "evidence": f"best OOD layer {int(ood_best.get('layer', -1))}; best high-loss layer {int(high_best.get('layer', -1))}",
            },
            {
                "hypothesis": "H4: Bayesian uncertainty adds signal",
                "verdict": "Not supported (diagonal Laplace approximation)",
                "evidence": "Laplace mean matches logistic; Laplace MI performs poorly; result conditional on diagonal posterior",
            },
        ]
    )

    qualitative_for_report = qualitative.copy()
    if not qualitative_for_report.empty:
        qualitative_for_report["_rank"] = 2
        qualitative_for_report.loc[
            (qualitative_for_report["case"] == "probe_catch")
            & (qualitative_for_report["set_name"] == "ood_char_shuffle"),
            "_rank",
        ] = 0
        qualitative_for_report.loc[qualitative_for_report["case"] == "probe_catch", "_rank"] = qualitative_for_report.loc[
            qualitative_for_report["case"] == "probe_catch", "_rank"
        ].clip(upper=1)
        qualitative_for_report = qualitative_for_report.sort_values(["_rank", "loss"], ascending=[True, False]).drop(columns=["_rank"])

    char_loss_note = ""
    if not ood_types.empty:
        char_probe = ood_types[(ood_types["ood_type"] == "char_shuffle") & (ood_types["method"] == "deterministic_logistic")]
        char_loss = ood_types[(ood_types["ood_type"] == "char_shuffle") & (ood_types["method"] == "loss")]
        if len(char_probe) and len(char_loss):
            char_loss_note = (
                f"On `char_shuffle`, next-token loss reaches {float(char_loss.iloc[0]['auroc']):.3f} AUROC, "
                f"above the layer-5 probe at {float(char_probe.iloc[0]['auroc']):.3f}. "
                "This suggests that the corruption is directly reflected in target-token surprise. "
                "However, loss requires knowledge of the correct next token and is therefore an oracle-style retrospective diagnostic rather than a generation-time monitor."
            )

    body = f"""# DOWSING: Detecting OOD and Weird States in NanoGPT

**White-box uncertainty monitoring in tiny transformers**

## Abstract

We train a small byte-level nanoGPT model on Tiny Shakespeare and test whether frozen internal activations contain signals predictive of high-loss and distribution-shifted tokens. We compare output-level confidence metrics against deterministic and diagonal-Laplace Bayesian probes trained on layerwise hidden states. The cleanest positive result is within-distribution: a held-out Shakespeare-only high-loss probe at layer {id_high_loss_probe_layer} reaches {id_high_loss_probe_auroc:.3f} AUROC vs entropy {id_high_loss_entropy_auroc:.3f}. The known-shift aggregate OOD probe reaches {ood_best.get('auroc', float('nan')):.3f} AUROC vs entropy {ood_entropy:.3f} but generalizes to only {loso_probe_beats_entropy} of {loso_total} held-out shift families under leave-one-shift-out, failing notably on char_shuffle. On the held-out shifts the logistic probe handles worst, an OOD-free Mahalanobis detector on early-layer activations generalizes better. Diagonal-Laplace Bayesian uncertainty does not improve over the deterministic logistic readout in this implementation, conditional on the diagonal posterior approximation.

## Hypotheses

H1: Layerwise activation probes predict high-loss tokens better than output entropy.
H2: Layerwise activation probes detect some OOD shifts better than output entropy.
H3: Probe performance varies by layer.
H4: Bayesian uncertainty provides additional signal beyond deterministic probe scores.

## Method

The harness trains a 6-layer byte-level nanoGPT model on Tiny Shakespeare, reserves held-out Shakespeare for ID evaluation, and builds synthetic Python, modern prose, shuffled Shakespeare, and repetition OOD sets. During evaluation it records per-token loss, entropy, max probability, target probability, model top prediction, and every block activation. Probes are trained on frozen post-block residual activations with sequence-level splits, so adjacent tokens from the same fixed window do not leak across train/test splits.

High-loss labels are defined by the top 20% of held-out ID losses. Target probability is therefore an oracle diagnostic for high-loss, because token loss is `-log(target probability)`. It is excluded from fair high-loss baseline comparisons.

The aggregate OOD probe is a known-shift detector: training includes examples from the same OOD families used at test time. To separate memorization of known shifts from generalization to unseen shifts, the robustness section adds a leave-one-OOD-type-out evaluation. The Python, modern prose, and repetition sets are template-generated and repeated to fill the eval split, so those rows should be read as engineering stress tests rather than natural distribution estimates. The controlled `char_shuffle` and `word_shuffle` sets are more load-bearing because they are derived from held-out Shakespeare bytes rather than tiled snippets.

## Run Metadata

- Compute: Modal NVIDIA L4 GPU
- Training: 3,000 iterations, batch size 64, block size 128
- Model: 6 layers, 6 heads, 192 embedding dimensions, byte vocabulary size 256
- Evaluation: 50,000 target tokens per eval set
- Final logged validation loss: approximately 1.444
- Artifact bundle: `modal_outputs/full-l4-3000-robustness-dowsing-results.tar.gz`

## Results

### Known-Shift Result

When the probe is trained on examples from all five OOD families, the best layer probe reached {ood_best.get('auroc', float('nan')):.3f} AUROC, compared with entropy at {ood_entropy:.3f} AUROC and loss at {ood_loss:.3f} AUROC. The improvement over entropy is {format_delta(ood_best.get('auroc', float('nan')) - ood_entropy)} AUROC. This is a useful known-shift monitoring result, not a claim of general OOD detection.

For the original high-loss task evaluated over ID plus OOD rows, the best layer probe reached {high_best.get('auroc', float('nan')):.3f} AUROC, compared with entropy at {high_entropy:.3f} AUROC and negative max probability at {high_neg_max:.3f} AUROC. This number is partly redundant with OOD detection because OOD rows are systematically higher loss; the ID-only high-loss robustness check below is the cleaner token-surprise test.

### Hypothesis Verdicts

{markdown_table(hypothesis_rows, ["hypothesis", "verdict", "evidence"])}

### Best OOD Layerwise Results

{markdown_table(ood, [c for c in ["task", "probe_type", "layer", "auroc", "auprc", "brier", "ece"] if c in ood.columns])}

### Best High-Loss Layerwise Results

{markdown_table(high_loss, [c for c in ["task", "probe_type", "layer", "auroc", "auprc", "brier", "ece"] if c in high_loss.columns])}

### Fair Baseline Comparison

{markdown_table(fair, [c for c in ["task", "method", "layer", "auroc", "auprc", "brier", "ece", "note"] if c in fair.columns])}

Calibration metrics are reported only for probability outputs. Raw distances, norms, margins, entropy, and loss are ranked scores, so their Brier, NLL, and ECE entries are intentionally blank.

Because logistic regression is trained with class weighting, its probabilities should be interpreted as scoring outputs rather than deployment-calibrated forecasts. AUROC and AUPRC are the primary metrics; calibration metrics are secondary diagnostics only.

### Robustness: Leave-One-OOD-Type-Out

{markdown_table(loso_summary, [c for c in ["holdout_ood_type", "best_probe_layer", "probe_auroc", "entropy_auroc", "loss_auroc", "delta_vs_entropy"] if c in loso_summary.columns])}

This evaluation trains the probe on ID plus four OOD families and tests it on the held-out fifth family plus held-out ID. It is a stricter test than the aggregate known-shift result because the target shift type is not present in probe training. The trained logistic probe beats entropy on {loso_probe_beats_entropy} of {loso_total} held-out shifts. The exception is `char_shuffle`, where the probe underperforms entropy ({format_delta(loso_char_shuffle_delta)} AUROC). This is the most damaging row in the report: char_shuffle was previously presented as the most controlled sanity check, and it is the one shift that the trained probe does not generalize to when held out of training. The known-shift char_shuffle gain therefore reflects in-distribution memorization of that shift family rather than a general OOD signal.

### Robustness: Best Detector Per Held-Out Shift

{markdown_table(loso_best_detector, [c for c in ["holdout_ood_type", "best_method", "best_layer", "best_auroc"] if c in loso_best_detector.columns])}

This table ranks only deployable detectors. The `loss` and `neg_target_prob` baselines are excluded because they require knowing the true next token; they appear in the LOSO summary above for reference but are oracle diagnostics, not monitors.

When the held-out shift is genuinely unseen, the trained logistic probe is not always the best deployable detector. Mahalanobis distance from the ID training mean — a one-class density estimator that is *not* trained on any OOD examples — beats the logistic probe on the two LOSO settings where the probe struggles most. On char_shuffle, Mahalanobis at layer 0 reaches 0.696 AUROC vs probe 0.590; on python, Mahalanobis at layer 1 reaches 0.858 vs probe 0.747. This suggests that for unseen shift types, an OOD-free density model on early-layer activations generalizes better than a trained linear readout, and that the right detector depends on the shift.

### Robustness: ID-Only High-Loss

{markdown_table(id_high_loss_summary, [c for c in ["method", "layer", "auroc", "auprc", "brier", "ece", "n_test"] if c in id_high_loss_summary.columns])}

This evaluation trains and tests only on held-out Shakespeare windows. The high-loss threshold is fit from ID train rows only, so the task measures within-distribution token surprise rather than dataset shift.

### OOD Type Breakdown

{markdown_table(ood_type_summary, [c for c in ["ood_type", "layer5_probe_auroc", "entropy_auroc", "delta"] if c in ood_type_summary.columns])}

The OOD-type breakdown tests whether the aggregate known-shift result is driven only by easy shifts such as Python code or repetition. Within the known-shift task, the layer-5 probe beats entropy on every shift, including controlled shuffled-Shakespeare. However, this section is **conditional on having seen each shift family during probe training**: the leave-one-shift-out section above shows that the char_shuffle gain in particular does not survive when char_shuffle is withheld from training. Read these per-type numbers as known-shift monitoring strength, not as evidence of general OOD detection.

The table reports the layer-5 probe for each OOD type, matching the best aggregate OOD layer. A separate layer-by-type sweep shows that Python and modern prose are most detectable in earlier or middle layers, while repetition, word shuffle, and char shuffle are strongest in later layers. This suggests that different distribution shifts become linearly accessible at different depths.

### Best Layer By OOD Type

{markdown_table(best_ood_type_layers, [c for c in ["ood_type", "best_layer", "best_probe_auroc"] if c in best_ood_type_layers.columns])}

{char_loss_note}

## Key Figures

- `artifacts/plots/ood_auroc_by_layer.png`
- `artifacts/plots/high_loss_auroc_by_layer.png`
- `artifacts/plots/ood_type_breakdown.png`
- `artifacts/plots/ood_probe_delta_by_type.png`
- `artifacts/plots/ood_layer_type_heatmap.png`
- `artifacts/plots/confident_failures_probe_score.png`

## Qualitative Examples

{markdown_table(qualitative_for_report.head(10), [c for c in ["case", "set_name", "target", "top_prediction", "loss", "entropy", "probe_score"] if c in qualitative_for_report.columns])}

The char-shuffle example shows the kind of confident-failure token output entropy misses. Important caveat: the probe used here was trained on examples from `ood_char_shuffle`, so this is an in-distribution catch under the known-shift task, not a generalization result. The leave-one-shift-out evaluation above shows the trained probe does not generalize to char_shuffle when that family is held out of training; under that stricter setup, layer-0 Mahalanobis is the better detector for this kind of corruption. This example illustrates the failure-mode geometry, not white-box monitoring of unseen shifts.

## Interpretation

The defensible claims are narrower than the known-shift headline suggests.

**Within-distribution token surprise (H1).** The cleanest result is the ID-only high-loss check. Trained on held-out Shakespeare windows alone, with the high-loss threshold fit from ID train rows only, the layer-{id_high_loss_probe_layer} logistic probe reaches {id_high_loss_probe_auroc:.3f} AUROC vs entropy {id_high_loss_entropy_auroc:.3f}. This is a real signal about token-level surprise that is not confounded with dataset shift, and it survives the audit corrections.

**Known-shift OOD (H2, in-distribution).** Trained on examples from all five OOD families, the layer-{int(ood_best.get('layer', -1))} probe reaches {ood_best.get('auroc', float('nan')):.3f} AUROC vs entropy {ood_entropy:.3f}. This is a useful known-shift monitoring result. It is not evidence of a general OOD signal: the probe has seen examples of every shift family during training.

**Leave-one-shift-out OOD (H2, generalization).** The trained probe beats entropy on {loso_probe_beats_entropy} of {loso_total} held-out shifts. It fails on char_shuffle ({format_delta(loso_char_shuffle_delta)} AUROC vs entropy), which had been presented as the most controlled sanity check. On char_shuffle and python — the two hardest LOSO settings for the logistic probe — Mahalanobis distance from the ID-training mean (a one-class detector that uses no OOD examples) generalizes better than the trained probe. The honest summary is that activations carry shift-relevant information, but which detector exposes it depends on whether the shift family was seen in training. The known-shift gain on char_shuffle reflects memorization of that family, not a general signal.

**Bayesian uncertainty (H4).** Negative for the specific approximation tested: diagonal-Laplace predictive mean is effectively identical to the deterministic logistic readout, and Laplace MI performs near chance. Residual-stream features are correlated, and a diagonal posterior with fixed prior precision is a severe approximation, so this is a result about *this approximation*, not about Bayesian uncertainty more generally. A full or low-rank posterior could change the picture.

**Caveats.** The probes are correlational; the model is tiny, undertrained (validation loss ~1.444), and run with a single LM seed and a single split seed; three of the OOD sets are tiled templates that admit content memorization across train/test windows. There are no error bars. The strongest claim the data support is: in this tiny-transformer setting, frozen activations carry within-distribution token-surprise signal beyond entropy, and carry known-shift detection signal that mostly but not fully generalizes to unseen shift families.

## Limitations

- Tiny model
- Byte-level tokenization
- Synthetic OOD distributions
- Python, prose, and repetition OOD sets are tiled templates, so they permit content memorization
- Aggregate OOD detection is a known-shift task unless using the leave-one-type-out robustness table
- Original high-loss detection is partly confounded with OOD; ID-only high-loss is cleaner
- Approximate diagonal Bayesian probe with fixed prior precision
- Single LM seed and single split seed; no error bars or bootstrap confidence intervals
- Post-block residual hooks only; no embedding or final layer-norm probe
- Position-in-window effects are not controlled
- Probes are correlational, not causal

## Summary Claim

DOWSING shows that, in this tiny-transformer setting, frozen nanoGPT activations carry two kinds of signal beyond output entropy: a within-distribution token-surprise signal (ID-only high-loss probe layer {id_high_loss_probe_layer}, {id_high_loss_probe_auroc:.3f} AUROC vs entropy {id_high_loss_entropy_auroc:.3f}), and a known-shift detection signal that generalizes to {loso_probe_beats_entropy} of {loso_total} held-out shift families but not to controlled char-shuffle when that family is withheld from training. On the held-out shifts where the trained probe struggles, an OOD-free Mahalanobis density estimator on early-layer activations is the better detector, suggesting the right white-box monitor is shift-dependent. Diagonal-Laplace Bayesian uncertainty did not add signal beyond the deterministic readout in this implementation, but this is contingent on the diagonal posterior approximation rather than a general statement about Bayesian uncertainty.

## Future Work

- Activation interventions along probe directions
- Replace tiled synthetic OOD with sampled corpora and de-duplicate windows across splits
- Report leave-one-shift-out as the primary OOD metric
- Evaluate high-loss within ID and within each OOD family separately
- Add bootstrap confidence intervals and repeat over language-model and split seeds
- Larger datasets and GPT-2 BPE variants
- Comparison against sparse autoencoder features
- Generation-time monitoring
"""
    report_path.write_text(body, encoding="utf-8")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
