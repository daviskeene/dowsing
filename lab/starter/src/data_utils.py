import json
from pathlib import Path

import numpy as np


def read_bin(path: str | Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.uint16)


def load_manifest(dataset_dir: str | Path) -> dict:
    with (Path(dataset_dir) / "manifest.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def make_windows(data: np.ndarray, block_size: int, stride: int) -> list[np.ndarray]:
    windows = []
    # Need block_size + 1 tokens so idx and targets both have block_size positions.
    for start in range(0, max(0, len(data) - block_size), stride):
        window = data[start : start + block_size + 1]
        if len(window) == block_size + 1:
            windows.append(window)
    return windows


def sequence_split(sequence_ids: np.ndarray, train_frac: float = 0.60, val_frac: float = 0.20, seed: int = 1337):
    unique = np.array(sorted(np.unique(sequence_ids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_train = int(len(unique) * train_frac)
    n_val = int(len(unique) * val_frac)
    train_ids = set(unique[:n_train])
    val_ids = set(unique[n_train : n_train + n_val])
    test_ids = set(unique[n_train + n_val :])
    return {
        "train": np.array([x in train_ids for x in sequence_ids]),
        "val": np.array([x in val_ids for x in sequence_ids]),
        "test": np.array([x in test_ids for x in sequence_ids]),
    }

