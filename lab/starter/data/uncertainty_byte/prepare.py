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
    """Encode text as UTF-8 bytes stored in nanoGPT-compatible uint16 IDs."""
    # TODO[DOWSING-01]: Turn Python text into byte token IDs.
    # This lab uses a byte tokenizer: UTF-8 bytes are the vocabulary, and each
    # byte value 0..255 becomes one token. Store the result as uint16 because
    # nanoGPT's dataset loader expects that dtype.
    raise NotImplementedError


def decode(ids: np.ndarray) -> str:
    """Decode byte IDs back into text for debugging."""
    # TODO[DOWSING-01]: Reverse encode() for quick sanity checks.
    # Cast IDs back to uint8 bytes, then decode as UTF-8. Use replacement for
    # invalid byte sequences so debugging never crashes on odd model outputs.
    raise NotImplementedError


def download_tiny_shakespeare(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    print(f"downloading Tiny Shakespeare to {path}")
    urllib.request.urlretrieve(DATA_URL, path)


def split_text(text: str, tiny: bool) -> tuple[str, str, str]:
    """Return LM train, LM val, and held-out ID eval splits."""
    # TODO[DOWSING-01]: Make the three text splits used by the rest of the lab.
    # If --tiny was passed, first keep a small prefix for fast smoke tests.
    # Then split by character position: 80% trains the language model, 10%
    # validates it, and the final 10% stays held out as clean in-distribution
    # evaluation text.
    raise NotImplementedError


def char_shuffle(text: str, window: int = 64, seed: int = 13) -> str:
    """Shuffle characters within fixed windows."""
    # TODO[DOWSING-01]: Create a simple OOD set by scrambling local character order.
    # Shuffle each fixed-size window independently with a seeded random.Random so
    # outputs are reproducible while the overall length stays unchanged.
    raise NotImplementedError


def word_shuffle(text: str, window: int = 48, seed: int = 17) -> str:
    """Shuffle word order within fixed windows."""
    # TODO[DOWSING-01]: Create another OOD set by scrambling word order.
    # Split on whitespace, shuffle words within each window, then join the
    # shuffled windows. The text remains word-like but should be less predictable.
    raise NotImplementedError


def synthetic_python(target_len: int) -> str:
    snippets = [
        "def normalize(values):\n    total = sum(values) or 1\n    return [v / total for v in values]\n",
        "class Cache:\n    def __init__(self):\n        self.items = {}\n\n    def get(self, key, default=None):\n        return self.items.get(key, default)\n",
        "for epoch in range(3):\n    loss = model(batch).mean()\n    optimizer.zero_grad()\n    loss.backward()\n    optimizer.step()\n",
    ]
    text = "\n".join(snippets)
    return (text * ((target_len // len(text)) + 1))[:target_len]


def modern_prose(target_len: int) -> str:
    paragraphs = [
        "The train arrived late, but nobody seemed surprised.",
        "At the office, the meeting moved quickly from budget questions to product risks.",
        "She opened the laptop at the kitchen table and read through the notes again.",
    ]
    text = "\n\n".join(paragraphs)
    return (text * ((target_len // len(text)) + 1))[:target_len]


def repetition(target_len: int) -> str:
    phrase = "To be or not to be. never never never.\n"
    return (phrase * ((target_len // len(phrase)) + 1))[:target_len]


def save_bin(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # TODO[DOWSING-01]: Write a nanoGPT .bin file.
    # Reuse encode(text), then call ndarray.tofile(path). Downstream code reads
    # these bytes back as uint16 token IDs.
    raise NotImplementedError


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

    # TODO[DOWSING-01]: Write meta.pkl for nanoGPT.
    # Include vocab_size=256 and tokenizer="byte"; simple stoi/itos mappings are
    # useful too, even though byte IDs already define the vocabulary.
    #
    # TODO[DOWSING-01]: Write manifest.json for later lab scripts.
    # It should name the dataset/tokenizer/vocab size and map split names to the
    # relative .bin paths written above.
    raise NotImplementedError


if __name__ == "__main__":
    main()
