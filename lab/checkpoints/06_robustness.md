# Checkpoint 06: robustness evals

## Goal

Separate known-shift detection from stricter generalization checks.

## Starter file

```text
lab/starter/scripts/robustness_evals.py
```

## Main concepts

- leave-one-OOD-type-out splitting
- ID-only high-loss labels
- threshold fitting from train rows only
- deployable detectors vs oracle diagnostics

## Check command

Fast function-level test:

```bash
.venv/bin/python lab/test.py --milestone 6
```

After copying your completed file:

```bash
.venv/bin/python scripts/robustness_evals.py --laplace_samples 16
```

## Expected outputs

```text
artifacts/results/ood_leave_one_type_out_summary.csv
artifacts/results/high_loss_id_clean_only_summary.csv
```

## Questions

- Why is aggregate OOD detection easier than leave-one-shift-out?
- Which held-out shift is the hardest for the logistic probe?
- Why is ID-only high-loss the cleaner high-loss result?

## Completion criteria

You can explain why the audited DOWSING claim is narrower than the original headline.
