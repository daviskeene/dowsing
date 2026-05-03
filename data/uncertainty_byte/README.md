# `uncertainty_byte` dataset

This directory contains the byte-level Tiny Shakespeare dataset used by DOWSING.

Tracked file:

```text
prepare.py
```

Generated files:

```text
train.bin
val.bin
meta.pkl
manifest.json
eval/*.bin
raw/tinyshakespeare.txt
```

The generated files are ignored by git. Recreate them with:

```bash
make prepare
```

For a quick tiny version:

```bash
python data/uncertainty_byte/prepare.py --tiny
```

