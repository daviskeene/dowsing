# Checkpoint 02: activation capture

## Goal

Capture per-token output metrics and post-block residual activations.

## Starter files

```text
lab/starter/src/activation_capture.py
lab/starter/scripts/collect_activations.py
```

## Main concepts

- PyTorch forward hooks
- logits, targets, and per-token cross entropy
- entropy and max probability
- target probability as an oracle diagnostic
- activation array row alignment with `metrics.csv`

## Check command

Fast function-level test:

```bash
.venv/bin/python lab/test.py --milestone 2
```

After copying completed starter files into the root implementation:

```bash
.venv/bin/python scripts/collect_activations.py \
  --ckpt out-uncertainty-byte/ckpt.pt \
  --dataset data/uncertainty_byte \
  --max_tokens_per_set 512 \
  --batch_size 4 \
  --device cpu
```

## Expected outputs

```text
artifacts/activations/id_clean/metrics.csv
artifacts/activations/id_clean/layer_00.npy
artifacts/activations/id_clean/layer_05.npy
```

## Questions

- What shape is each `layer_XX.npy` file?
- Why does the collector drop the first few positions?
- What activation site does the hook capture?

## Completion criteria

You can collect activations and verify that metric rows match activation rows.
