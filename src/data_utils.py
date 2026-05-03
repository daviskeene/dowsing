import json
from pathlib import Path

import numpy as np


OOD_TYPES = {
    "id_clean": "id_clean",
    "ood_python": "python",
    "ood_modern_prose": "modern_prose",
    "ood_char_shuffle": "char_shuffle",
    "ood_word_shuffle": "word_shuffle",
    "ood_repetition": "repetition",
}


def encode(text: str) -> np.ndarray:
    return np.frombuffer(text.encode("utf-8", errors="replace"), dtype=np.uint8).astype(np.uint16)


def decode(ids: np.ndarray) -> str:
    return bytes(ids.astype(np.uint8).tolist()).decode("utf-8", errors="replace")


def read_tokens(path: str | Path) -> np.ndarray:
    return np.memmap(path, dtype=np.uint16, mode="r")


def load_manifest(dataset_dir: str | Path) -> dict:
    with (Path(dataset_dir) / "manifest.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def eval_split_paths(dataset_dir: str | Path) -> dict[str, Path]:
    dataset_dir = Path(dataset_dir)
    manifest = load_manifest(dataset_dir)
    return {
        name: dataset_dir / rel
        for name, rel in manifest["splits"].items()
        if name.startswith("id_") or name.startswith("ood_")
    }


def sequence_split(sequence_ids: np.ndarray, train_frac: float = 0.6, val_frac: float = 0.2, seed: int = 1337) -> dict[str, np.ndarray]:
    unique = np.unique(sequence_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_train = int(len(unique) * train_frac)
    n_val = int(len(unique) * val_frac)
    split_ids = {
        "train": unique[:n_train],
        "val": unique[n_train : n_train + n_val],
        "test": unique[n_train + n_val :],
    }
    # Split by source sequence so near-duplicate windows do not leak across train/val/test.
    return {split: np.isin(sequence_ids, ids) for split, ids in split_ids.items()}


def iter_token_windows(tokens: np.ndarray, block_size: int, stride: int):
    # The extra token supplies the next-token target for the final input position.
    need = block_size + 1
    for start in range(0, max(0, len(tokens) - need + 1), stride):
        chunk = np.asarray(tokens[start : start + need], dtype=np.int64)
        if len(chunk) == need:
            yield start, chunk[:-1], chunk[1:]
