# Checkpoint 01: byte-level data prep

## Goal

Implement the data preparation contract.

## Starter file

```text
lab/starter/data/uncertainty_byte/prepare.py
```

## Main concepts

- byte-level tokenization
- train/validation/held-out ID split
- synthetic OOD vs controlled OOD
- nanoGPT `meta.pkl`
- `manifest.json`

## Check command

Fast function-level test:

```bash
.venv/bin/python lab/test.py --milestone 1
```

After copying your completed starter file into `data/uncertainty_byte/prepare.py`:

```bash
.venv/bin/python data/uncertainty_byte/prepare.py --tiny
```

## Expected outputs

```text
data/uncertainty_byte/train.bin
data/uncertainty_byte/val.bin
data/uncertainty_byte/meta.pkl
data/uncertainty_byte/manifest.json
data/uncertainty_byte/eval/id_clean.bin
data/uncertainty_byte/eval/ood_char_shuffle.bin
data/uncertainty_byte/eval/ood_word_shuffle.bin
```

## Questions

- Why does byte-level encoding use `np.uint16` instead of `np.uint8` for nanoGPT files?
- Which OOD sets are most vulnerable to memorization?
- Why are char shuffle and word shuffle more controlled than Python snippets?

## Completion criteria

You can prepare the tiny dataset and explain every file in `data/uncertainty_byte/`.
