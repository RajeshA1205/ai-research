"""Hyperparameter configuration for the Transformer.

Every component of the model is parameterised by a handful of integers:
embedding size, number of heads, number of layers, etc. Rather than passing
those around as a long list of arguments (or, worse, hard-coding them), we
collect them into a single ``ModelConfig`` dataclass that gets handed to the
top-level ``Transformer`` constructor. The model then reads what it needs.

The defaults in this file are deliberately **smaller than the paper's**:

    ===========  ==========  ============================
    Hyperparam   Paper       This file (M4 16GB-friendly)
    ===========  ==========  ============================
    d_model      512         256
    d_ff         2048        1024
    n_layers     6           6
    n_heads      8           8
    ===========  ==========  ============================

Halving ``d_model`` quarters the size of every Q/K/V projection and roughly
quarters activation memory. That makes the difference between "fits in 16 GB
of unified memory while doing something else" and "swap thrashing." When you
move on to a real translation task, bump these back up — but for the toy
copy task and unit tests, the smaller numbers are plenty.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """All hyperparameters needed to build a Transformer.

    Using ``@dataclass`` gives us:
      * ``__init__`` for free (matching the field declarations below)
      * ``__repr__`` for free (helpful for the "cfg=..." print at training time)
      * Field-by-field equality (handy in tests)

    Attributes:
        src_vocab: Number of distinct token IDs the **source** sequence can use
            (e.g., the size of the English vocabulary in EN→DE translation).
            Determines the size of the source embedding matrix.
        tgt_vocab: Same idea for the **target** sequence vocabulary. We allow
            different sizes for src and tgt because real translation systems
            usually have separate vocabularies.
        d_model: The model's "main" embedding dimension. Every token, once
            embedded, lives as a ``d_model``-dimensional vector, and that
            same dimension is used as the residual-stream width throughout
            the encoder and decoder. Paper: 512. We use 256 for memory.
        n_heads: Number of parallel attention heads. The paper splits
            ``d_model`` evenly into ``n_heads`` chunks of size
            ``d_k = d_model / n_heads`` (32 with our defaults). More heads
            = more independent "viewpoints" of the same sequence, but each
            individual head sees a lower-dimensional subspace.
        n_layers: Number of stacked encoder layers (and the same number of
            decoder layers). Paper uses 6.
        d_ff: Hidden width of the position-wise feed-forward sub-layer
            (Linear(d_model→d_ff) → ReLU → Linear(d_ff→d_model)). Paper uses
            2048; we use 1024. Roughly 4× ``d_model`` is the conventional
            ratio.
        dropout: Dropout probability applied at several points (after
            attention softmax, after the FFN's ReLU, after the residual
            sum, on the positional encodings). Paper uses 0.1.
        max_len: Maximum sequence length the positional encoding table will
            be precomputed for. Sequences longer than this will index past
            the buffer and crash — bump this up if you need longer context.
            Decoupled from ``n_layers`` and ``d_model``; only memory cost
            is one (max_len, d_model) tensor, which is tiny.
        pad_id: Integer token ID reserved for padding. The masking utilities
            and label-smoothing loss both use this to ignore pad positions.
            Conventional choice is 0; we follow that here.
    """

    # No defaults for vocab sizes because every dataset is different. Forcing
    # the caller to specify them avoids silently building a model with the
    # wrong-sized embedding tables.
    src_vocab: int
    tgt_vocab: int

    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    max_len: int = 512
    pad_id: int = 0

    def __post_init__(self) -> None:
        """Sanity-check that the multi-head split is well-defined.

        ``MultiHeadAttention`` reshapes a tensor from ``(B, S, d_model)`` into
        ``(B, n_heads, S, d_k)`` where ``d_k = d_model // n_heads``. That
        reshape only makes sense if ``d_model`` is exactly divisible by
        ``n_heads``. Catching this here gives a clear error message at config
        time, not a cryptic ``view`` failure deep inside attention.
        """
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )

    @property
    def d_k(self) -> int:
        """Per-head key/query/value dimension.

        Computed property (not a stored field) so it can never drift out of
        sync with ``d_model`` and ``n_heads``. Used inside the attention
        modules when reshaping to/from per-head form.
        """
        return self.d_model // self.n_heads
