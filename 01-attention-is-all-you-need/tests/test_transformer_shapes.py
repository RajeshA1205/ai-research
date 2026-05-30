"""End-to-end forward / backward shape contracts for the full Transformer.

The point of these tests is not to verify training quality (that's what
scripts/train_copy_task.py is for) — only that the model is wired up
correctly:

  * Forward pass produces logits with shape (B, S_tgt, tgt_vocab).
  * Every parameter is reachable from the loss (gets a non-None grad
    after backward).

Together with the per-component tests, passing these means we have a
plumbing-correct Transformer; if loss doesn't go down at training time,
the bug is in the math/init/optimizer, not the wiring.
"""

from __future__ import annotations

import torch

from transformer import ModelConfig
from transformer.transformer import Transformer


def test_forward_output_shape() -> None:
    """Logits shape == (batch, target-seq-len, target-vocab)."""
    cfg = ModelConfig(
        src_vocab=20,
        tgt_vocab=20,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
        dropout=0.0,
        max_len=16,
    )
    model = Transformer(cfg).eval()  # eval() disables dropout — purely a shape test

    B, S_src, S_tgt = 2, 7, 5
    src = torch.randint(1, cfg.src_vocab, (B, S_src))
    tgt = torch.randint(1, cfg.tgt_vocab, (B, S_tgt))

    logits = model(src, tgt)
    assert logits.shape == (B, S_tgt, cfg.tgt_vocab)


def test_parameters_receive_gradients() -> None:
    """Every parameter is reachable from the loss.

    The loop at the bottom is the catch-all: ``p.grad is None`` means a
    parameter was registered (e.g. in __init__) but never used in
    ``forward()``. That's almost always a wiring bug.
    """
    cfg = ModelConfig(
        src_vocab=20,
        tgt_vocab=20,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
        dropout=0.0,
        max_len=16,
    )
    model = Transformer(cfg)
    src = torch.randint(1, cfg.src_vocab, (2, 7))
    tgt = torch.randint(1, cfg.tgt_vocab, (2, 5))
    logits = model(src, tgt)
    # Sum is enough — any scalar produces a valid backward graph.
    logits.sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
