"""Quick end-to-end forward+backward smoke test.

Run with:  uv run python scripts/sanity_check.py

What this script is for
-----------------------
Once you've implemented all the modules, this is the first thing to run.
It checks the easy-to-test invariants of the whole pipeline before you
sink time into a training run:

  1. The model can be moved to MPS (or CPU) without errors.
  2. A forward pass produces output with the expected shape ``(B, S, V)``.
  3. ``.backward()`` runs without exploding or hitting an op that's not
     supported on the chosen device.
  4. Every parameter received a gradient. (If anything is unreachable from
     the loss — e.g. a sub-module that was registered but never used in
     ``forward()`` — its grad will still be ``None`` and we catch that
     here, not after a 30-minute training run.)

If this script prints "forward ok" and "backward ok" cleanly, the model
is plumbed correctly. Anything that's wrong with the actual training
dynamics (loss not going down, etc.) is a separate concern handled by
``train_copy_task.py``.
"""

from __future__ import annotations

import torch

from transformer import ModelConfig, get_device
from transformer.transformer import Transformer


def main() -> None:
    # 1) Pick a device. ``get_device()`` returns MPS on M-series Macs.
    device = get_device()
    print(f"Using device: {device}")

    # 2) Build a tiny model. Using small dims (d_model=64, n_layers=2) so
    #    the script finishes in well under a second even on CPU. The point
    #    is to test the *shape* of the pipeline, not anything quantitative.
    cfg = ModelConfig(
        src_vocab=32,
        tgt_vocab=32,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.0,
        max_len=32,
    )
    # Move the whole thing to the chosen device. ``.to`` recursively walks
    # all submodules / parameters / buffers (including the positional
    # encoding table).
    model = Transformer(cfg).to(device)

    # 3) Random input batch. Tokens drawn from [1, vocab) so we don't
    #    accidentally generate pad (id 0). Shape (B, S).
    B, S = 2, 10
    src = torch.randint(1, cfg.src_vocab, (B, S), device=device)
    tgt = torch.randint(1, cfg.tgt_vocab, (B, S), device=device)

    # 4) Forward pass. We pass no masks — fine for a sanity check, since
    #    no positions are actually padding.
    logits = model(src, tgt)
    # Shape contract: logits should be (batch, target-seq-len, target-vocab).
    assert logits.shape == (B, S, cfg.tgt_vocab), logits.shape
    print(f"forward ok: logits {tuple(logits.shape)}")

    # 5) Dummy loss (just a sum) and backward. Sum is enough — a real loss
    #    would be a label-smoothed KL or cross-entropy, but for testing
    #    the autograd graph any scalar works.
    loss = logits.sum()
    loss.backward()

    # 6) Check that every parameter got a gradient. ``p.grad is None`` means
    #    the parameter was never used in the forward pass — usually a wiring
    #    bug (e.g. a module registered in __init__ but accidentally bypassed
    #    in forward). We list any offenders rather than just asserting, so
    #    you can see exactly which parameter is unreachable.
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    if missing:
        raise RuntimeError(f"params with no grad: {missing}")
    print(f"backward ok: {sum(p.numel() for p in model.parameters()):,} params")


if __name__ == "__main__":
    main()
