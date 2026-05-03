# DOWSING fill-in lab

This directory is a guided version of the DOWSING codebase.

The main repo is the complete reference implementation. The `lab/starter/` tree mirrors the important parts of that implementation, but several functions are intentionally left incomplete. The goal is to rebuild the pipeline piece by piece and use the checkpoints to test your understanding.

## How to use this lab

Work through the exercises in order:

1. `data/uncertainty_byte/prepare.py`
2. `src/activation_capture.py`
3. `src/metrics.py`
4. `src/probes.py`
5. `src/bayes_laplace.py`
6. `scripts/collect_activations.py`
7. `scripts/train_probes.py`
8. `scripts/robustness_evals.py`

Each starter file contains `TODO[DOWSING-*]` markers. Fill those in, then use the matching checkpoint in `lab/checkpoints/`.

You can run the lab test bench at any point:

```bash
python lab/test.py --milestone 1
python lab/test.py --milestone 2
python lab/test.py --all
```

The tests import files from `lab/starter/`, so they check your fill-in definitions without touching the finished root implementation.

The starter files are not wired into the main `Makefile`. That is deliberate: the root repo stays runnable. To test a completed exercise, either:

- compare your solution against the matching root file, or
- copy your completed starter file into the corresponding root path on a throwaway branch and run the checkpoint command.

Example:

```bash
cp lab/starter/src/metrics.py src/metrics.py
.venv/bin/python -m py_compile src/metrics.py
```

If you are doing this in a real repo, use git before copying files around:

```bash
git checkout -b lab-metrics-exercise
```

## Recommended workflow

For each exercise:

1. Read the checkpoint first.
2. Open the starter file.
3. Fill only the TODO sections.
4. Run the checkpoint command.
5. Compare your completed file with the root implementation.
6. Write down what the function is doing in plain English.

The last step matters. The point is not just to make the code pass. The point is to understand what each piece contributes to the monitoring experiment.

## Checkpoint index

| checkpoint | topic | main files |
| --- | --- | --- |
| 00 | setup and repo map | `README.md`, `LAB.md`, `Makefile` |
| 01 | byte-level data prep | `data/uncertainty_byte/prepare.py` |
| 02 | activation capture | `src/activation_capture.py`, `scripts/collect_activations.py` |
| 03 | metrics and baselines | `src/metrics.py`, `scripts/eval_probes.py` |
| 04 | deterministic probes | `src/probes.py`, `scripts/train_probes.py` |
| 05 | diagonal Laplace probe | `src/bayes_laplace.py` |
| 06 | robustness evals | `scripts/robustness_evals.py` |
| 07 | interpretation | `report/mini_report.md`, `artifacts/results/` |

## Test bench

`lab/test.py` contains small synthetic tests for the starter functions. It is meant to be fast and local, not a substitute for the full experiment.

Milestones:

```text
1  byte-level data prep
2  activation capture
3  metrics
4  deterministic probes
5  diagonal Laplace probe
6  robustness split helpers
```

Examples:

```bash
python lab/test.py --milestone 3
python lab/test.py --all
```

If a milestone fails with `NotImplementedError`, keep filling in the TODOs for that milestone. If it fails with a value mismatch, the test output should point to the contract that is not being met.

## Completion target

By the end, you should be able to explain:

- how bytes become nanoGPT tokens
- where activations are captured
- how per-token confidence metrics are computed
- what a linear probe learns
- why high-loss target probability is an oracle diagnostic
- why known-shift OOD is easier than leave-one-shift-out OOD
- why the Bayesian result is conditional on the diagonal approximation
- what result DOWSING actually supports
