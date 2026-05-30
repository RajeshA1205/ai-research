"""Stack of N decoder layers, with a final LayerNorm on the way out.

Paper section: §3.1 ("Encoder and Decoder Stacks"), N=6 in the paper.

Mirror image of ``encoder.py`` — same construction (``nn.ModuleList`` of
sub-layers + a final ``LayerNorm``), but each sub-layer is a ``DecoderLayer``
which has the extra cross-attention sub-layer. The decoder also threads the
encoder ``memory`` along with the running decoder state.
"""

from __future__ import annotations

from torch import Tensor, nn

from .layernorm import LayerNorm
from .layers import DecoderLayer


class Decoder(nn.Module):
    """N stacked decoder layers + final LayerNorm.

    Args mirror the encoder; see ``encoder.py`` for explanations.
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
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        # Final norm — same reasoning as in the encoder.
        self.norm = LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ) -> Tensor:
        """Run the full decoder.

        Args:
            x: ``(B, S_tgt, d_model)`` — embedded target sequence (already
                has positional encodings added).
            memory: ``(B, S_src, d_model)`` — encoder output. Same memory is
                fed to every decoder layer's cross-attention.
            src_mask: ``(B, 1, 1, S_src)`` padding mask for the source
                (used inside cross-attention).
            tgt_mask: ``(B, 1, S_tgt, S_tgt)`` combined pad+causal mask for
                the target (used inside masked self-attention).

        Returns:
            ``(B, S_tgt, d_model)`` — the decoder's final state, ready to
            be projected to vocabulary logits by the output head.
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
