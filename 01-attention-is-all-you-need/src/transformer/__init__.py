"""Top-level package for the from-scratch Transformer.

This file is the public face of the ``transformer`` package. When you write
``import transformer`` in another file, Python runs this module first. We use
it for two things:

  1. Re-export the most commonly needed names so callers can do
     ``from transformer import ModelConfig, get_device`` without having to
     remember the internal file layout (``transformer.config.ModelConfig``,
     etc.). Anything not re-exported here is still importable via its full
     submodule path — re-exporting is a convenience, not a wall.

  2. Provide a small ``get_device()`` helper. Every script in this project
     starts by asking "where do I run — MPS or CPU?", and we don't want that
     decision duplicated in every file.

The Transformer architecture itself is split across the sibling files:

    config.py            : ModelConfig dataclass (hyperparameters)
    masking.py           : padding + causal attention masks
    layernorm.py         : hand-rolled LayerNorm
    positional.py        : sinusoidal positional encoding
    attention.py         : scaled-dot-product + multi-head attention
    feedforward.py       : position-wise feed-forward sub-layer
    layers.py            : SublayerConnection, EncoderLayer, DecoderLayer
    encoder.py           : N-layer encoder stack
    decoder.py           : N-layer decoder stack
    transformer.py       : full Transformer (embeddings + encoder + decoder + output proj)
    label_smoothing.py   : label-smoothed KL-divergence loss (paper §5.4)
    optim.py             : Noam learning-rate scheduler (paper §5.3)
"""

from __future__ import annotations

import torch

# We re-export ``ModelConfig`` so that ``from transformer import ModelConfig``
# works without the user having to know it lives in ``config.py``.
from .config import ModelConfig

# ``__all__`` is the explicit "public API" of this package. Anything listed
# here is what we promise as the stable, intended-for-export surface. (It also
# controls what ``from transformer import *`` would expose, though we don't
# encourage star-imports.)
__all__ = ["ModelConfig", "get_device"]


def get_device() -> torch.device:
    """Return the best available compute device on this machine.

    Priority: Apple's MPS backend (the GPU on M-series Macs) → CPU.

    On the M4 we always want MPS — it's roughly an order of magnitude faster
    than CPU for the matmul-heavy ops in a Transformer. We don't list CUDA
    because the target hardware is a Mac; if you ever run this on a CUDA box,
    add ``torch.cuda.is_available()`` as a higher-priority branch.

    Returns:
        A ``torch.device`` you can pass to ``.to(device)`` on tensors and
        modules. Most code in this repo just does
        ``device = get_device(); model = model.to(device)``.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
