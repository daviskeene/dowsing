import argparse
import csv
from pathlib import Path
import sys

for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists() and (parent / "model.py").exists():
        sys.path.insert(0, str(parent))
        break

import numpy as np
import torch

from src.activation_capture import per_token_metrics, register_block_hooks, remove_hooks
from src.data_utils import load_manifest, make_windows, read_bin
from src.nanogpt_loader import load_nanogpt_checkpoint


OOD_TYPE = {
    "id_clean": "id_clean",
    "ood_python": "python",
    "ood_modern_prose": "modern_prose",
    "ood_char_shuffle": "char_shuffle",
    "ood_word_shuffle": "word_shuffle",
    "ood_repetition": "repetition",
}


def collect_set(
    model,
    data: np.ndarray,
    set_name: str,
    out_dir: Path,
    block_size: int,
    stride: int,
    batch_size: int,
    max_tokens: int,
    drop_first_k: int,
    device: str,
) -> None:
    """Collect metrics.csv and layer_XX.npy for one eval set."""
    out_dir.mkdir(parents=True, exist_ok=True)
    windows = make_windows(data, block_size=block_size, stride=stride)
    activations, handles = register_block_hooks(model)
    metrics_rows = []
    layer_chunks: dict[int, list[np.ndarray]] = {i: [] for i in range(len(model.transformer.h))}
    row_id = 0

    try:
        for start in range(0, len(windows), batch_size):
            batch_windows = windows[start : start + batch_size]
            if not batch_windows:
                continue

            # TODO[DOWSING-02]: Build next-token prediction tensors.
            # Each window has block_size + 1 token IDs. idx is all but the last
            # token, targets is all but the first token, so position t in idx is
            # trained/evaluated against the next token in targets.
            raise NotImplementedError

            with torch.no_grad():
                logits, _ = model(idx, targets)
                token_metrics = per_token_metrics(logits, targets)

            keep = slice(drop_first_k, None)
            kept = targets[:, keep].numel()

            # TODO[DOWSING-02]: Save the kept hidden states for every layer.
            # Hook outputs have shape [B,T,C]. Apply the same position slice used
            # for metrics, move to CPU, and append rows shaped [kept_tokens, C] to
            # layer_chunks[layer]. Keeping rows aligned with metrics.csv is crucial.
            raise NotImplementedError

            # TODO[DOWSING-02]: Append one metrics row per kept token.
            # Include row_id, set_name, is_ood, ood_type, sequence_id, position,
            # token_id, target_id, loss, entropy, max_prob, target_prob,
            # top_pred_id, and top_pred_prob. row_id should increase once per
            # token and match the row order of the saved layer arrays.
            raise NotImplementedError

            if row_id >= max_tokens:
                break
    finally:
        remove_hooks(handles)

    # TODO[DOWSING-02]: Write this eval set's artifacts.
    # Save metrics_rows as metrics.csv. For each layer, concatenate its chunks
    # and write layer_XX.npy. Empty outputs should still be valid arrays so later
    # scripts fail clearly instead of on a missing file.
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="out-uncertainty-byte/ckpt.pt")
    parser.add_argument("--dataset", default="data/uncertainty_byte")
    parser.add_argument("--out_dir", default="artifacts/activations")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_tokens_per_set", type=int, default=50_000)
    parser.add_argument("--drop_first_k_positions", type=int, default=8)
    args = parser.parse_args()

    model = load_nanogpt_checkpoint(args.ckpt, args.device)
    manifest = load_manifest(args.dataset)

    for set_name, rel_path in manifest["splits"].items():
        if set_name in {"lm_train", "lm_val"}:
            continue
        data = read_bin(Path(args.dataset) / rel_path)
        collect_set(
            model=model,
            data=data,
            set_name=set_name,
            out_dir=Path(args.out_dir) / set_name,
            block_size=args.block_size,
            stride=args.stride,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens_per_set,
            drop_first_k=args.drop_first_k_positions,
            device=args.device,
        )


if __name__ == "__main__":
    main()
