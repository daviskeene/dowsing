import argparse
import json
import pickle
import random
import urllib.request
from pathlib import Path

import numpy as np


DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
ROOT = Path(__file__).resolve().parent


def encode(text: str) -> np.ndarray:
    return np.frombuffer(text.encode("utf-8", errors="replace"), dtype=np.uint8).astype(np.uint16)


def decode(ids: np.ndarray) -> str:
    return bytes(ids.astype(np.uint8).tolist()).decode("utf-8", errors="replace")


def download_tiny_shakespeare(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    print(f"downloading Tiny Shakespeare to {path}")
    urllib.request.urlretrieve(DATA_URL, path)


def split_text(text: str, tiny: bool) -> tuple[str, str, str]:
    if tiny:
        text = text[:60_000]
    n = len(text)
    train_end = int(n * 0.80)
    val_end = int(n * 0.90)
    return text[:train_end], text[train_end:val_end], text[val_end:]


def char_shuffle(text: str, window: int = 64, seed: int = 13) -> str:
    rng = random.Random(seed)
    chunks = []
    for i in range(0, len(text), window):
        chars = list(text[i : i + window])
        rng.shuffle(chars)
        chunks.append("".join(chars))
    return "".join(chunks)


def word_shuffle(text: str, window: int = 48, seed: int = 17) -> str:
    rng = random.Random(seed)
    words = text.split()
    chunks = []
    for i in range(0, len(words), window):
        group = words[i : i + window]
        rng.shuffle(group)
        chunks.append(" ".join(group))
    return "\n".join(chunks)


def synthetic_python(target_len: int) -> str:
    snippets = [
        "def normalize(values):\n    total = sum(values) or 1\n    return [v / total for v in values]\n",
        "class Cache:\n    def __init__(self):\n        self.items = {}\n\n    def get(self, key, default=None):\n        return self.items.get(key, default)\n",
        "for epoch in range(3):\n    loss = model(batch).mean()\n    optimizer.zero_grad()\n    loss.backward()\n    optimizer.step()\n",
        "with open(path, 'r', encoding='utf-8') as handle:\n    rows = [line.strip().split(',') for line in handle]\n",
    ]
    text = "\n".join(snippets)
    return (text * ((target_len // len(text)) + 1))[:target_len]


def modern_prose(target_len: int) -> str:
    paragraphs = [
        "The train arrived late, but nobody seemed surprised. People checked their phones, shifted their bags, and waited for the platform sign to change.",
        "At the office, the meeting moved quickly from budget questions to product risks. The team wanted a clear decision before the end of the week.",
        "She opened the laptop at the kitchen table and read through the notes again. The problem was not complicated, but the details mattered.",
        "By evening the city had cooled down. Restaurants filled slowly, buses hissed at the curb, and the streets reflected the last light.",
    ]
    text = "\n\n".join(paragraphs)
    return (text * ((target_len // len(text)) + 1))[:target_len]


def repetition(target_len: int) -> str:
    phrase = "To be or not to be. never never never.\n"
    return (phrase * ((target_len // len(phrase)) + 1))[:target_len]


def save_bin(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encode(text).tofile(path)
    print(f"wrote {path} ({path.stat().st_size // 2:,} tokens)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiny", action="store_true", help="prepare a small subset for smoke tests")
    args = parser.parse_args()

    raw_path = ROOT / "raw" / "tinyshakespeare.txt"
    download_tiny_shakespeare(raw_path)
    text = raw_path.read_text(encoding="utf-8")
    lm_train, lm_val, id_eval = split_text(text, args.tiny)
    target_len = max(len(id_eval), 4096)

    save_bin(ROOT / "train.bin", lm_train)
    save_bin(ROOT / "val.bin", lm_val)

    eval_sets = {
        "id_clean": id_eval,
        "ood_python": synthetic_python(target_len),
        "ood_modern_prose": modern_prose(target_len),
        "ood_char_shuffle": char_shuffle(id_eval),
        "ood_word_shuffle": word_shuffle(id_eval),
        "ood_repetition": repetition(target_len),
    }
    for name, eval_text in eval_sets.items():
        save_bin(ROOT / "eval" / f"{name}.bin", eval_text)

    meta = {
        "vocab_size": 256,
        "tokenizer": "byte",
        "itos": {i: chr(i) for i in range(256)},
        "stoi": {chr(i): i for i in range(256)},
    }
    with (ROOT / "meta.pkl").open("wb") as f:
        pickle.dump(meta, f)

    manifest = {
        "dataset": "uncertainty_byte",
        "tokenizer": "byte",
        "vocab_size": 256,
        "splits": {
            "lm_train": "train.bin",
            "lm_val": "val.bin",
            **{name: f"eval/{name}.bin" for name in eval_sets},
        },
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {ROOT / 'meta.pkl'} and {ROOT / 'manifest.json'}")


if __name__ == "__main__":
    main()
