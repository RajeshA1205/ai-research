"""Attention masks: padding mask + subsequent (causal) mask.

Paper section: §3.2.3 ("Applications of Attention in our Model").

Why we need masks at all
------------------------
Attention computes ``softmax(QKᵀ / √d_k) · V``. The softmax is over the *key*
axis, meaning every query position gets a probability distribution over every
key position. Sometimes we want to **forbid** certain (query, key) pairs from
contributing — softmax should put zero weight on them. Two situations come up:

1. **Padding**: when we batch sentences of different lengths, shorter ones
   get padded out with a special PAD token. Those positions carry no real
   information, so no query — anywhere in the batch — should attend to them.

2. **Causality** (decoder self-attention only): during training, the decoder
   sees the entire target sequence at once (teacher forcing). To prevent it
   from "cheating" by looking at future tokens it hasn't generated yet, we
   forbid each position ``i`` from attending to positions ``j > i``.

How masks work mechanically
---------------------------
The convention used everywhere in this codebase: ``True`` (or ``1``) means
"keep this attention edge"; ``False`` (or ``0``) means "kill it". Inside
``scaled_dot_product_attention`` the mask is consumed by

    scores = scores.masked_fill(mask == 0, float("-inf"))

so killed positions get ``-inf`` raw scores, which become ``0`` after softmax.

Mask shapes (broadcast-friendly)
--------------------------------
Attention scores have shape ``(B, h, S_q, S_k)`` (batch, heads, queries, keys).
We build masks with singleton (size-1) dimensions where they don't depend on
that axis, so PyTorch's broadcasting auto-expands them at no memory cost:

    pad mask     : (B, 1, 1, S_k)   — same across heads & queries
    causal mask  : (1, 1, S_q, S_k) — same across batch & heads
    combined tgt : (B, 1, S_q, S_k) — pad varies per-batch, causal stays fixed
"""

from __future__ import annotations

import torch
from torch import Tensor


def make_pad_mask(seq: Tensor, pad_id: int = 0) -> Tensor:
    """Build a key-side padding mask from a batch of token IDs.

    Args:
        seq: ``(B, S)`` integer tensor of token IDs. The same tensor we later
            feed into the embedding layer.
        pad_id: The integer token ID that means "this position is padding".
            Paper convention (and ours) is 0.

    Returns:
        ``(B, 1, 1, S)`` bool tensor. Positions equal to ``pad_id`` are
        ``False`` ("kill"); all other positions are ``True`` ("keep").

    The two singleton dims are intentional:
      * dim 1 (heads): the same token is masked for every head.
      * dim 2 (query positions): every query in the sequence sees the same
        set of keys, so the mask doesn't depend on which query is asking.

    Example::

        seq = [[3, 7, 2, 0, 0]]   # last two positions are padding
        make_pad_mask(seq) shape (1, 1, 1, 5), values [[[[T, T, T, F, F]]]]
    """
    # ``seq != pad_id`` is an elementwise compare → bool tensor of shape (B, S).
    # ``unsqueeze(1)`` → (B, 1, S); a second ``unsqueeze(2)`` → (B, 1, 1, S).
    return (seq != pad_id).unsqueeze(1).unsqueeze(2)


def make_causal_mask(size: int, device: torch.device | None = None) -> Tensor:
    """Build a square lower-triangular mask preventing peeking at the future.

    Args:
        size: The sequence length ``S``. Mask will be ``(S, S)`` before
            adding broadcast dims.
        device: Optional device to build the tensor on. Pass the model's
            device so we don't pay a host→device copy each forward pass.

    Returns:
        ``(1, 1, size, size)`` bool tensor where entry ``[..., i, j]`` is
        ``True`` iff ``j ≤ i``. Inside attention, query position ``i`` can
        only attend to key positions ``0..i``.

    Visual (size=4)::

        T F F F
        T T F F
        T T T F
        T T T T

    The two singleton dims at the front (batch, heads) let this mask
    broadcast across any batch size and any number of heads.
    """
    # ``torch.tril`` zeros out everything strictly above the main diagonal.
    # We start from a tensor of all ones (so tril keeps lower-triangle as 1)
    # and use ``dtype=torch.bool`` so the result is a boolean mask directly.
    mask = torch.tril(torch.ones(size, size, dtype=torch.bool, device=device))
    # Add broadcast dims for batch and heads.
    return mask.unsqueeze(0).unsqueeze(0)


def make_tgt_mask(tgt: Tensor, pad_id: int = 0) -> Tensor:
    """Build the combined mask used by the decoder's self-attention.

    The decoder needs BOTH:
      * Don't attend to padding positions in the (batched) target.
      * Don't attend to future positions (causality).

    Args:
        tgt: ``(B, S_tgt)`` integer tensor — the decoder input (typically
            ``[BOS, y1, y2, ...]``).
        pad_id: Padding token ID.

    Returns:
        ``(B, 1, S_tgt, S_tgt)`` bool mask. Position ``[b, 0, i, j]`` is
        ``True`` iff target token ``j`` is not padding AND ``j ≤ i``.

    Implementation note: we logically AND the pad mask with the causal mask.
    Broadcasting handles the shape gymnastics:

        pad_mask  : (B, 1, 1,     S_tgt)
        causal    : (1, 1, S_tgt, S_tgt)
        AND result: (B, 1, S_tgt, S_tgt)
    """
    # Pad mask: shape (B, 1, 1, S_tgt) — only depends on key position.
    pad_mask = make_pad_mask(tgt, pad_id)
    # Causal mask: shape (1, 1, S_tgt, S_tgt) — only depends on (q, k) pair.
    # We pass tgt.device so the mask is created directly on the right device
    # (avoids an MPS↔CPU copy every forward pass).
    causal = make_causal_mask(tgt.size(1), tgt.device)
    # Bitwise AND combines the two; broadcasting fills in the missing dims.
    # An (i, j) edge survives only if BOTH masks say "keep".
    return pad_mask & causal
