from pathlib import Path
import sys

import torch


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def load_nanogpt_checkpoint(ckpt_path: str, device: str = "cpu"):
    _ensure_repo_on_path()
    from model import GPT, GPTConfig

    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = checkpoint["model_args"]
    model = GPT(GPTConfig(**model_args))
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            # torch.compile can add this wrapper prefix; the plain model class expects bare keys.
            state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
