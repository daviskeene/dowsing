# Checkpoint 03: metrics and baselines

## Goal

Implement binary ranking and calibration metrics.

## Starter file

```text
lab/starter/src/metrics.py
```

## Main concepts

- AUROC
- AUPRC
- Brier score
- negative log likelihood
- expected calibration error
- recall at fixed FPR
- probability outputs vs raw ranking scores

## Check command

Fast function-level test:

```bash
.venv/bin/python lab/test.py --milestone 3
```

After copying your completed file into `src/metrics.py`:

```bash
.venv/bin/python -m py_compile src/metrics.py
.venv/bin/python scripts/eval_probes.py --task ood
```

## Questions

- Why can AUROC be computed on raw scores?
- Why should Brier/NLL/ECE be blank for Mahalanobis distance?
- What happens when `y_true` has only one class?

## Completion criteria

You can explain which metrics require calibrated probabilities and which only require ranked scores.
