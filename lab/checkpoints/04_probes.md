# Checkpoint 04: deterministic probes

## Goal

Train deterministic probe baselines over frozen activations.

## Starter files

```text
lab/starter/src/probes.py
lab/starter/scripts/train_probes.py
```

## Main concepts

- feature standardization
- sequence-level splitting
- logistic regression
- ridge classifier
- Mahalanobis distance
- standardized activation norm

## Check command

Fast function-level test:

```bash
.venv/bin/python lab/test.py --milestone 4
```

After copying completed starter files:

```bash
.venv/bin/python scripts/train_probes.py --task ood --max_train_tokens 512 --laplace_samples 16
```

## Expected outputs

```text
artifacts/probes/ood/layer_00.pkl
artifacts/results/ood_layerwise.csv
```

## Questions

- Why split by sequence instead of token?
- Why standardize activations using training rows only?
- What is Mahalanobis distance measuring in this setup?

## Completion criteria

You can train OOD probes and identify the best layer by AUROC.
