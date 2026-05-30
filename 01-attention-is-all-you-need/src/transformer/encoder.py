"""Stack of N encoder layers, with a final LayerNorm on the way out.

Paper section: §3.1 ("Encoder and Decoder Stacks"), N=6 in the paper.

This module is intentionally minimal — all the work happens in ``EncoderLayer``
(see ``layers.py``). Here we just glue ``n_layers`` of them in a row and
apply one extra ``LayerNorm`` at the end.

Why the extra LayerNorm at the end?
-----------------------------------
With the **pre-norm** convention we use (see ``layers.py``), each sub-layer
normalises its *input*. That means the very last sub-layer's output is the
unnormalised post-residual sum, which can drift in scale. Applying one more
LayerNorm after the stack puts the encoder's external interface back into a
predictable scale before it gets handed to the decoder's cross-attention.
"""

from __future__ import annotations

from torch import Tensor, nn

from .layernorm import LayerNorm
from .layers import EncoderLayer


class Encoder(nn.Module):
    """N stacked encoder layers + final LayerNorm.

    Args:
        n_layers: How many ``EncoderLayer``s to stack. Paper uses 6.
        d_model: Width of the residual stream (same for every layer).
        n_heads: Number of attention heads per layer.
        d_ff: Feed-forward hidden dim per layer.
        dropout: Dropout probability inside every layer.
    """

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        # ``nn.ModuleList`` (rather than a plain Python list) is important —
        # it tells PyTorch about the contained modules so they show up in
        # ``.parameters()``, ``.to(device)``, ``state_dict()``, etc. Each
        # layer is a *separate* module with its own weights.
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        # Final norm applied once after the whole stack (see module docstring).
        self.norm = LayerNorm(d_model)

    def forward(self, x: Tensor, src_mask: Tensor | None = None) -> Tensor:
        """Run the full encoder.

        Args:
            x: ``(B, S_src, d_model)`` — embedded source sequence (already
                has positional encodings added).
            src_mask: ``(B, 1, 1, S_src)`` padding mask — the same mask is
                reused at every layer.

        Returns:
            ``(B, S_src, d_model)`` — the encoder "memory" that the decoder
            will cross-attend to.
        """
        # Threading the state through each layer in turn. Each layer reads
        # the previous layer's output and produces an updated representation.
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)
