# DOWSING lab guide

This guide turns DOWSING into a reproducible mini-lab for learning white-box monitoring on a small transformer.

The goal is to run the experiment, inspect each stage, understand the result, and then choose a follow-up direction. The report explains what happened in the completed run. This file explains how to walk through the lab yourself.

If you want a fill-in version of the code, use `lab/`. It mirrors the main repo at the key learning points and leaves selected functions incomplete with `TODO[DOWSING-*]` markers. The root repo is the reference implementation; `lab/` is the workbook.

The workbook also has a small test bench:

```bash
python lab/test.py --milestone 1
python lab/test.py --all
```

These tests validate the starter functions directly, so learners can get feedback before running the full pipeline.

## Learning outcomes

After completing the lab, you should be able to:

- Train a small byte-level nanoGPT model from scratch.
- Prepare ID and OOD evaluation sets for a language-model monitoring experiment.
- Capture post-block residual activations with PyTorch forward hooks.
- Compute token-level confidence metrics: loss, entropy, max probability, and target probability.
- Train layerwise linear probes over frozen activations.
- Compare probes against output-confidence and density-style baselines.
- Explain why target probability is an oracle diagnostic for high-loss labels.
- Distinguish known-shift OOD detection from leave-one-shift-out OOD generalization.
- Interpret a mixed result without overclaiming.
- Identify concrete extensions that would make the experiment more rigorous.

## Prerequisites

You should be comfortable with:

- Python scripts and virtual environments
- basic PyTorch tensors and model checkpoints
- logistic regression or linear classifiers
- AUROC as a ranking metric
- reading CSV outputs and simple matplotlib plots

You do not need prior nanoGPT experience. The relevant nanoGPT files are included in this repo.

## Compute options

There are two practical ways to run the lab.

### Option A: local CPU

Use this for smoke tests and code inspection.

Pros:

- no cloud setup
- enough to verify the full pipeline shape
- good for editing scripts and checking artifacts

Cons:

- the full 3,000-iteration training run can take hours
- activation collection and probe sweeps are slower

Recommended local command:

```bash
make smoke
```

### Option B: Modal GPU

Use this for the full experiment.

Pros:

- uses a cloud L4 GPU
- runs the full training and evaluation pipeline much faster
- downloads a compact result bundle automatically

Cons:

- requires a Modal account and CLI authentication
- raw activations are omitted from the default bundle unless requested

Recommended Modal command:

```bash
modal run modal_run.py --mode full --run-name full-l4-3000-robustness
```

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
make setup
```

This installs:

```text
torch
numpy
tqdm
scikit-learn
scipy
pandas
matplotlib
modal
```

## Modal setup

Authenticate once:

```bash
source .venv/bin/activate
modal setup
```

Modal opens a browser authentication flow and writes a token to your local Modal config. After that, the repo's `modal_run.py` can launch the remote pipeline.

Run a small GPU smoke test:

```bash
modal run modal_run.py --mode smoke --run-name smoke-modal
```

Run the full experiment:

```bash
modal run modal_run.py --mode full --run-name full-l4-3000-robustness
```

The Modal runner:

1. builds an image from `requirements.txt`
2. uploads the repo, excluding local artifacts and raw generated data
3. prepares the byte-level dataset remotely
4. trains nanoGPT on a Modal L4 GPU
5. collects activations
6. trains and evaluates probes
7. runs robustness checks
8. generates plots and `report/mini_report.md`
9. writes a tarball to a Modal Volume
10. downloads and extracts the tarball into `modal_outputs/`

By default, the downloaded tarball includes:

```text
README.md
LAB.md
SPEC.md
CONTEXT.md
config/train_uncertainty_byte.py
out-uncertainty-byte/ckpt.pt
artifacts/results/
artifacts/plots/
artifacts/probes/
report/
```

Raw activation arrays are larger and are not included by default. To include them:

```bash
modal run modal_run.py --mode full --include-activations
```

## Lab sequence

For the full repository layout, see `docs/PROJECT_STRUCTURE.md`.

### 1. Verify the pipeline

Run:

```bash
make smoke
```

What this does:

- prepares a tiny dataset
- trains a 10-iteration CPU checkpoint
- collects a small number of activations
- trains OOD and high-loss probes
- runs robustness checks
- generates plots and a report

Expected outputs:

```text
out-uncertainty-byte/ckpt.pt
artifacts/results/
artifacts/plots/
report/mini_report.md
```

Important: smoke-test numbers are not meaningful. This step only checks that the pipeline works.

Read next:

- `scripts/smoke_test.py`
- `Makefile`

Questions:

- Which commands make up the full pipeline?
- Which artifacts are generated?
- Which artifacts are intermediate files and which are final presentation outputs?

### 2. Prepare the byte-level dataset

Run:

```bash
make prepare
```

What this does:

- downloads Tiny Shakespeare
- splits text into LM train, LM validation, and held-out ID eval
- encodes text as UTF-8 bytes stored in `np.uint16`
- creates synthetic and controlled OOD eval sets
- writes nanoGPT-compatible `meta.pkl`
- writes `manifest.json`

Expected outputs:

```text
data/uncertainty_byte/train.bin
data/uncertainty_byte/val.bin
data/uncertainty_byte/meta.pkl
data/uncertainty_byte/manifest.json
data/uncertainty_byte/eval/id_clean.bin
data/uncertainty_byte/eval/ood_python.bin
data/uncertainty_byte/eval/ood_modern_prose.bin
data/uncertainty_byte/eval/ood_char_shuffle.bin
data/uncertainty_byte/eval/ood_word_shuffle.bin
data/uncertainty_byte/eval/ood_repetition.bin
```

Read next:

- `data/uncertainty_byte/prepare.py`
- `src/data_utils.py`

Questions:

- Why does byte-level tokenization avoid OOD unknown-character problems?
- Which OOD sets are controlled transformations of held-out Shakespeare?
- Which OOD sets are tiled templates and therefore weaker evidence?

### 3. Train the model

Local:

```bash
make train
```

Modal:

```bash
modal run modal_run.py --mode full --run-name full-l4-3000-robustness
```

What this does:

- trains a 6-layer, 6-head, 192-dim byte-level nanoGPT model
- saves a checkpoint in `out-uncertainty-byte/ckpt.pt`

Config:

```text
config/train_uncertainty_byte.py
```

Expected checkpoint:

```text
out-uncertainty-byte/ckpt.pt
```

Read next:

- `config/train_uncertainty_byte.py`
- `train.py`
- `model.py`

Questions:

- What model size is being trained?
- Why keep 6 layers but reduce width relative to the default Shakespeare config?
- What information is stored in the checkpoint?

### 4. Collect activations

Run:

```bash
make collect
```

What this does:

- loads the trained nanoGPT checkpoint
- runs fixed windows from each eval set through the frozen model
- records output metrics for each token
- saves post-block residual activations for every transformer block

Expected outputs:

```text
artifacts/activations/id_clean/metrics.csv
artifacts/activations/id_clean/layer_00.npy
artifacts/activations/id_clean/layer_01.npy
...
artifacts/activations/ood_word_shuffle/layer_05.npy
```

Read next:

- `scripts/collect_activations.py`
- `src/nanogpt_loader.py`
- `src/activation_capture.py`

Questions:

- What columns are stored in `metrics.csv`?
- Which activation site is captured?
- Why does the collector drop the first few positions in each context window?

### 5. Train and evaluate probes

Run:

```bash
make probes
make eval
```

What this does:

- trains probes for OOD detection
- trains probes for high-loss detection
- evaluates output-level baselines
- writes layerwise result tables

Probe types:

```text
deterministic_logistic
ridge_classifier
mahalanobis
activation_norm
laplace_predictive_mean
laplace_mutual_information
```

Expected outputs:

```text
artifacts/probes/ood/layer_00.pkl
artifacts/probes/high_loss/layer_00.pkl
artifacts/results/ood_layerwise.csv
artifacts/results/high_loss_layerwise.csv
artifacts/results/baselines.csv
```

Read next:

- `scripts/train_probes.py`
- `scripts/eval_probes.py`
- `src/probes.py`
- `src/bayes_laplace.py`
- `src/metrics.py`

Questions:

- Why split by sequence instead of token?
- Why is `target_prob` unfair for high-loss detection?
- Why are Brier, NLL, and ECE blank for raw scores like Mahalanobis distance?
- What does diagonal Laplace approximate?

### 6. Run robustness checks

Run:

```bash
make robustness
```

What this does:

- runs leave-one-OOD-type-out evaluation
- runs ID-only high-loss evaluation
- writes robustness result tables

Expected outputs:

```text
artifacts/results/ood_leave_one_type_out_layerwise.csv
artifacts/results/ood_leave_one_type_out_summary.csv
artifacts/results/high_loss_id_clean_only_layerwise.csv
artifacts/results/high_loss_id_clean_only_summary.csv
```

Read next:

- `scripts/robustness_evals.py`

Questions:

- Why is aggregate OOD detection a known-shift task?
- Which held-out OOD type breaks the logistic probe?
- Why is ID-only high-loss cleaner than high-loss over ID plus OOD?
- When does Mahalanobis beat the trained probe?

### 7. Make plots and report

Run:

```bash
make plots
make report
```

What this does:

- creates layerwise AUROC plots
- creates OOD type breakdown plots
- creates heatmaps and confident-failure plots
- generates the final Markdown report

Expected outputs:

```text
artifacts/plots/ood_auroc_by_layer.png
artifacts/plots/high_loss_auroc_by_layer.png
artifacts/plots/ood_type_breakdown.png
artifacts/plots/ood_probe_delta_by_type.png
artifacts/plots/ood_layer_type_heatmap.png
artifacts/plots/confident_failures_probe_score.png
report/mini_report.md
```

Read next:

- `scripts/make_plots.py`
- `scripts/analyze_results.py`
- `scripts/make_report_tables.py`
- `report/mini_report.md`

Questions:

- Which plot best shows layer localization?
- Which table gives the cleanest high-loss result?
- Which table gives the stricter OOD result?
- What is the strongest claim supported by the evidence?

## Result checklist

After a full run, verify these files exist:

```text
out-uncertainty-byte/ckpt.pt
artifacts/results/ood_layerwise.csv
artifacts/results/high_loss_layerwise.csv
artifacts/results/ood_leave_one_type_out_summary.csv
artifacts/results/high_loss_id_clean_only_summary.csv
artifacts/plots/ood_auroc_by_layer.png
artifacts/plots/high_loss_auroc_by_layer.png
artifacts/plots/ood_layer_type_heatmap.png
report/mini_report.md
```

## Current reference result

The current full Modal run produced:

```text
ID-only high-loss:
  layer-4 logistic probe: 0.779 AUROC
  entropy baseline:       0.656 AUROC

Known-shift aggregate OOD:
  layer-5 logistic probe: 0.879 AUROC
  entropy baseline:       0.566 AUROC

Leave-one-OOD-type-out:
  logistic probe beats entropy on 4 of 5 held-out shifts
  held-out char_shuffle is the failure case
```

The report contains the detailed tables and caveats.

## Extension exercises

These are listed roughly in order of "biggest research payoff" rather than "easiest to do." If you're trying to figure out where to start, the causal-direction extension is the one that connects DOWSING most directly to mechanistic interpretability and activation steering. It turns the probe from a passive readout into a tool for moving the model around. The cleanup-style extensions below it tighten the empirical claim.

### Causal probe directions

Use the learned probe direction as an intervention rather than only a readout.

Tasks:

- add or subtract the normalized probe direction at layer `L`
- measure changes in loss, entropy, and target probability
- compare intervention effects across layers

Main question:

> Is the probe only reading out a correlation, or does moving along the direction change model behavior?

This is the bridge from "monitoring" to "steering": the same direction that detects a state can sometimes induce it, and that connection is what makes probes interesting beyond classification. In the dowsing-rod metaphor: detection is the rod *twitching* over hidden trouble; steering is the rod *pulling*.

### More activation sites

Probe additional locations inside the transformer.

Tasks:

- capture token-plus-position embeddings
- capture pre-attention residual stream
- capture post-attention residual stream
- capture post-MLP residual stream
- capture final layernorm input

Main question:

> Where does failure information first become linearly accessible?

### Better OOD data

Replace tiled OOD templates with sampled corpora.

Tasks:

- sample many Python files rather than repeating a few snippets
- sample modern prose from a public-domain corpus
- de-duplicate byte windows across probe train/test splits
- make leave-one-shift-out the primary OOD metric

Main question:

> Do activation monitors generalize to genuinely new shift examples?

### Better high-loss evaluation

Evaluate high-loss within each distribution separately.

Tasks:

- run ID-only high-loss by position bin
- run high-loss within each OOD family
- compare probes with and without position features
- report threshold source explicitly

Main question:

> Are probes detecting token surprise, dataset identity, or position-in-window effects?

### Better Bayesian probes

Replace the diagonal-Laplace approximation.

Tasks:

- sweep prior precision
- implement full-covariance or low-rank Laplace for the linear probe
- compare against probe ensembles
- report MI AUROC and predictive-mean AUROC separately

Main question:

> Is the negative Bayesian result about Bayesian uncertainty, or about this approximation?
