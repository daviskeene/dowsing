# DOWSING lab spec: layerwise uncertainty probes for nanoGPT

Frozen nanoGPT activations contain layer-localized signals that predict high-loss and some distribution-shifted tokens better than output-level confidence metrics alone.

DOWSING is a mini AI safety lab: a small, inspectable environment for practicing white-box monitoring, uncertainty evaluation, distribution-shift testing, and honest empirical interpretation before moving to larger systems.

This repository implements a byte-level nanoGPT model trained on Tiny Shakespeare, activation capture over every transformer block, deterministic and diagonal-Laplace linear probes, output-confidence baselines, robustness checks, plots, and a generated mini-report.

The audited result is intentionally narrower than the original project hypothesis: ID-only high-loss detection is positive, known-shift OOD detection is strong, leave-one-shift-out OOD is mixed, and diagonal-Laplace Bayesian uncertainty does not improve over deterministic readouts in this implementation.

## Done Criteria

`make smoke` should prepare a tiny dataset, train a very short CPU checkpoint, collect activations, train and evaluate OOD probes, generate plots, and write `report/mini_report.md`.

For full experiments:

```bash
make setup
make prepare
make train
make collect
make probes
make eval
make robustness
make plots
make report
```
