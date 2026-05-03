# Checkpoint 00: setup and repo map

## Goal

Understand what each part of the repo does before filling in code.

## Commands

```bash
python3 -m venv .venv
source .venv/bin/activate
make setup
make smoke
```

## Expected outputs

```text
out-uncertainty-byte/ckpt.pt
artifacts/results/
artifacts/plots/
report/mini_report.md
```

## Questions

- Which commands does `make smoke` run?
- Which files define the model, data, probes, metrics, and report?
- Why are smoke-test numbers not meaningful?

## Completion criteria

You can explain the whole pipeline in one paragraph without looking at the code.

