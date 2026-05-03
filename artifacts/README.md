# Artifacts

This directory holds outputs from DOWSING runs.

Tracked by default:

```text
results/
plots/
```

Ignored by default:

```text
activations/
probes/
```

Activation arrays and serialized probes can be large and are easy to regenerate from a checkpoint. The lightweight CSV tables and plots are kept visible so the project can be inspected on GitHub.

