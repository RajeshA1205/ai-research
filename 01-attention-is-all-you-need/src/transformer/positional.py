"""Sinusoidal positional encoding.

Paper section: §3.5 ("Positional Encoding").

Why we need this at all
-----------------------
Self-attention is **permutation-equivariant**: shuffle the input tokens and
the output gets shuffled the same way, with no other change. The attention
mechanism, taken alone, has no idea whether token #3 came before or after
token #7. For language modelling we obviously need order, so we have to put
that information *somewhere*. The paper's solution: add a position-dependent
vector to each token embedding before the first encoder/decoder layer. The
combined vector encodes "what word + where it is".

Why sinusoidal (instead of learned) embeddings?
-----------------------------------------------
The paper compares both and reports near-identical performance. They picked
sinusoids because:

  * The encoding can be evaluated at any position without retraining, so the
    model can in principle generalise to sequences longer than anything seen
    at training time (the function is defined for all real positions).
  * Each pair of dimensions encodes position at a different "frequency",
    which makes relative offsets easy for the model to discover via simple
    linear combinations:  PE(pos+k) is a fixed linear function of PE(pos).

The exact formula (paper eq. just before §3.6) is

    PE[pos, 2i]   = sin( pos / 10000^(2i / d_model) )
    PE[pos, 2i+1] = cos( pos / 10000^(2i / d_model) )

Even-indexed dims get sin, odd-indexed dims get cos, and the wavelengths
form a geometric progression from 2π up to 10000·2π as ``i`` walks across
``d_model``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    """Non-learned sinusoidal position embedding, precomputed once.

    On construction we build a ``(max_len, d_model)`` table of position
    encodings, register it as a *buffer* (not a parameter — it has no
    gradients), and on every forward we slice out the first ``S`` rows and
    add them to the input.

    Buffers behave like parameters in two important ways:
      * They get moved automatically when you call ``.to(device)``.
      * They are included in ``state_dict()``, so saving/loading the model
        round-trips them.
    But they are NOT updated by the optimizer. Perfect for a constant table.
    """

    # Class-level annotation. Doesn't actually create the attribute (that
    # happens inside __init__ via register_buffer); it tells type-checkers
    # like mypy and IDE auto-complete that ``self.pe`` is a Tensor.
    pe: Tensor

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        """Precompute the ``(1, max_len, d_model)`` PE table.

        Args:
            d_model: Embedding dimension. The table will be ``d_model`` wide
                so it can be added to the embeddings directly.
            max_len: Largest sequence length we'll ever see. The table is
                only built once, so this just sets an upper bound. Memory
                cost is ``max_len * d_model`` floats — negligible.
            dropout: Dropout rate applied to ``embedding + PE`` (the paper
                does this).
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # ``position`` has shape (max_len, 1) and contains [0, 1, 2, ..., max_len-1]
        # as a column vector. The trailing size-1 dim is so it broadcasts cleanly
        # against ``div_term`` (shape (d_model/2,)) below.
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        # We want, for each pair (pos, i), the value
        #     pos / 10000^(2i / d_model)   =   pos * 10000^(-2i / d_model)
        # The naive way to compute 10000^(-2i / d_model) is `math.pow(10000, ...)`,
        # but that loses precision for large exponents. We use the identity
        #     10000^x = exp(x * ln(10000))
        # so the exponent is a linear scan and stays well-conditioned. ``div_term``
        # is the vector of these "frequency" factors, one per even-indexed dim,
        # with shape (d_model/2,).
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / d_model)
        )

        # Allocate the table and fill it. Multiplying (max_len, 1) by (d_model/2,)
        # broadcasts to (max_len, d_model/2) — one frequency per dim, all rows
        # advancing through positions.
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # even dims → sin
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims  → cos

        # Add a leading batch dim so we can use plain `+` against any
        # ``(B, S, d_model)`` input via broadcasting.  Shape: (1, max_len, d_model).
        # ``register_buffer`` makes ``self.pe`` an attribute that:
        #   * is NOT a learnable parameter (no grad)
        #   * IS moved by .to(device)
        #   * IS saved/loaded with state_dict
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        """Add positional info to embeddings, then dropout.

        Args:
            x: ``(B, S, d_model)`` token embeddings. Must already be the
                output of an embedding layer (and conventionally scaled by
                ``sqrt(d_model)`` — see ``transformer.Embeddings``).

        Returns:
            ``(B, S, d_model)`` — same shape, now with position info baked
            into each vector.
        """
        # ``self.pe[:, : x.size(1)]`` slices the first S rows of the precomputed
        # table — works whether S is shorter than max_len or equal to it.
        # Broadcast-add against (B, S, d_model).
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
