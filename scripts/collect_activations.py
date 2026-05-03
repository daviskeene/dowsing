import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.activation_capture import register_block_hooks, remove_hooks, token_metrics
from src.data_utils import OOD_TYPES, eval_split_paths, iter_token_windows, read_tokens
from src.nanogpt_loader import load_nanogpt_checkpoint


def infer_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def collect_set(model, set_name: str, bin_path: Path, out_dir: Path, args) -> None:
    tokens = read_tokens(bin_path)
    out_set = out_dir / set_name
    out_set.mkdir(parents=True, exist_ok=True)
    block_size = min(args.block_size or model.config.block_size, model.config.block_size)
    windows = []
    token_budget = 0
    for sequence_id, (start, x, y) in enumerate(iter_token_windows(tokens, block_size, args.stride)):
        # Early positions have little context, so probes train on the more stable
        # later tokens from each window.
        keep = max(0, block_size - args.drop_first_k_positions)
        if keep == 0:
            continue
        windows.append((sequence_id, start, x, y))
        token_budget += keep
        if token_budget >= args.max_tokens_per_set:
            break

    layer_chunks = {layer: [] for layer in range(model.config.n_layer)}
    rows = []
    row_id = 0
    # Hooked block activations and metrics.csv rows share the same row order; later
    # scripts rely on that alignment when loading layer_XX.npy files.
    activations, handles = register_block_hooks(model)
    try:
        for batch_start in tqdm(range(0, len(windows), args.batch_size), desc=set_name):
            batch = windows[batch_start : batch_start + args.batch_size]
            x_np = np.stack([item[2] for item in batch])
            y_np = np.stack([item[3] for item in batch])
            x = torch.from_numpy(x_np).long().to(args.device)
            y = torch.from_numpy(y_np).long().to(args.device)
            activations.clear()
            with torch.no_grad():
                logits, _ = model(x, y)
                metrics = token_metrics(logits.float(), y)
                probs = torch.softmax(logits.float(), dim=-1)
                top_prob, top_pred = probs.max(dim=-1)

            pos = np.arange(block_size)
            keep_mask = pos >= args.drop_first_k_positions
            remaining = args.max_tokens_per_set - row_id
            if remaining <= 0:
                break
            flat_keep = np.tile(keep_mask, len(batch))
            keep_indices = np.where(flat_keep)[0][:remaining]

            for layer in range(model.config.n_layer):
                act = activations[layer].detach().float().cpu().numpy().reshape(-1, model.config.n_embd)
                layer_chunks[layer].append(act[keep_indices].astype(np.float16))

            metric_np = {k: v.detach().float().cpu().numpy().reshape(-1) for k, v in metrics.items()}
            top_pred_np = top_pred.detach().cpu().numpy().reshape(-1)
            top_prob_np = top_prob.detach().float().cpu().numpy().reshape(-1)
            x_flat = x_np.reshape(-1)
            y_flat = y_np.reshape(-1)
            for local_flat in keep_indices:
                batch_item = local_flat // block_size
                position = local_flat % block_size
                sequence_id, start, _, _ = batch[batch_item]
                rows.append(
                    {
                        "row_id": row_id,
                        "set_name": set_name,
                        "is_ood": int(set_name != "id_clean"),
                        "ood_type": OOD_TYPES.get(set_name, set_name.replace("ood_", "")),
                        "sequence_id": int(sequence_id),
                        "source_start": int(start),
                        "position": int(position),
                        "token_id": int(x_flat[local_flat]),
                        "target_id": int(y_flat[local_flat]),
                        "top_pred_id": int(top_pred_np[local_flat]),
                        "top_pred_prob": float(top_prob_np[local_flat]),
                        "loss": float(metric_np["loss"][local_flat]),
                        "entropy": float(metric_np["entropy"][local_flat]),
                        "max_prob": float(metric_np["max_prob"][local_flat]),
                        "target_prob": float(metric_np["target_prob"][local_flat]),
                    }
                )
                row_id += 1
    finally:
        remove_hooks(handles)

    pd.DataFrame(rows).to_csv(out_set / "metrics.csv", index=False)
    for layer, chunks in layer_chunks.items():
        arr = np.concatenate(chunks, axis=0) if chunks else np.empty((0, model.config.n_embd), dtype=np.float16)
        np.save(out_set / f"layer_{layer:02d}.npy", arr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="out-uncertainty-byte/ckpt.pt")
    parser.add_argument("--dataset", default="data/uncertainty_byte")
    parser.add_argument("--out_dir", default="artifacts/activations")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_tokens_per_set", type=int, default=50_000)
    parser.add_argument("--drop_first_k_positions", type=int, default=8)
    args = parser.parse_args()
    args.device = infer_device(args.device)

    model = load_nanogpt_checkpoint(args.ckpt, args.device)
    out_dir = Path(args.out_dir)
    for set_name, path in eval_split_paths(args.dataset).items():
        collect_set(model, set_name, path, out_dir, args)


if __name__ == "__main__":
    main()
