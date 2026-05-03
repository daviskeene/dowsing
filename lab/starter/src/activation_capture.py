import torch
import torch.nn.functional as F


def register_block_hooks(model):
    """Register hooks on each transformer block and return (activations, handles)."""
    activations = {}
    handles = []

    # TODO[DOWSING-02]: Attach one forward hook to each transformer block.
    # Iterate over model.transformer.h with enumerate(). Each hook should save
    # output.detach() in activations[layer_idx] so probe training can read hidden
    # states without keeping the language-model computation graph alive.
    # Store every returned hook handle in handles; remove_hooks() uses them later.
    raise NotImplementedError

    return activations, handles


def remove_hooks(handles) -> None:
    for handle in handles:
        handle.remove()


def per_token_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute per-token loss and confidence metrics from logits [B,T,V]."""
    # TODO[DOWSING-02]: Convert logits into per-token uncertainty signals.
    # Compute log_probs/probs over the vocab dimension, cross-entropy loss for
    # the actual next token, entropy of the full distribution, max_prob, and the
    # probability assigned to targets. Also return the top predicted token ID and
    # its probability. Every returned tensor should have shape [B,T].
    raise NotImplementedError
