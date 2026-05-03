from __future__ import annotations

import torch
from torch.nn import functional as F


def register_block_hooks(model):
    activations = {}
    handles = []
    for layer_idx, block in enumerate(model.transformer.h):
        def hook(module, inputs, output, layer_idx=layer_idx):
            # Probe training reads these hidden states after the model's normal forward pass.
            activations[layer_idx] = output.detach()

        handles.append(block.register_forward_hook(hook))
    return activations, handles


def remove_hooks(handles) -> None:
    for handle in handles:
        handle.remove()


def token_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).view_as(targets)
    entropy = -(probs * log_probs).sum(dim=-1)
    max_prob = probs.max(dim=-1).values
    # The probability assigned to the actual next token is a compact confidence signal.
    target_prob = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return {
        "loss": loss,
        "entropy": entropy,
        "max_prob": max_prob,
        "target_prob": target_prob,
    }
