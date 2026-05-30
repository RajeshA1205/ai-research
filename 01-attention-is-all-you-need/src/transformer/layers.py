"""Encoder and decoder layers + the residual sub-layer wrapper.

Paper section: §3.1 ("Encoder and Decoder Stacks") — and Figure 1, which
this file is essentially the code translation of.

The pattern: every sub-layer is wrapped in residual + norm
----------------------------------------------------------
The paper says each sub-layer's output is

    LayerNorm( x + Sublayer(x) )                # paper, eq. before §3.1

i.e. **post-norm**: do the sub-layer, add the residual, then normalise.
That's what's in the diagram. We deviate slightly:

    x + Dropout( Sublayer( LayerNorm(x) ) )     # this codebase

This is **pre-norm** — normalise *before* the sub-layer, not after. It's
become the de-facto standard since 2019 because:

  * Pre-norm Transformers train reliably without the elaborate learning-rate
    warmup schedule the paper needed (the Noam scheduler in optim.py is
    still implemented for fidelity, but it's less critical).
  * Gradients flow more directly through the residual path, so very deep
    stacks behave better.

Both variants give similar final accuracy. Switching back to post-norm is a
one-line change in ``SublayerConnection.forward`` if you want to follow the
paper to the letter.
"""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor, nn

from .attention import MultiHeadAttention
from .feedforward import PositionwiseFeedForward
from .layernorm import LayerNorm


class SublayerConnection(nn.Module):
    """Residual + LayerNorm + dropout wrapper applied around any sub-layer.

    Pre-norm formulation:

        y = x + Dropout( sublayer( LayerNorm(x) ) )

    The sub-layer is supplied as a callable at forward time, so this one
    class works for both attention and feed-forward sub-layers (and could
    handle any future sub-layer with the same I/O shape).

    Args:
        features: The width of the residual stream (== ``d_model``). Needed
            so the internal LayerNorm can size its gamma/beta correctly.
        dropout: Dropout probability applied to the sub-layer's output
            before adding it to the residual.
    """

    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        # Each SublayerConnection has its own LayerNorm — they're not shared
        # across sub-layers. (Sharing would be wrong: each has its own input
        # distribution.)
        self.norm = LayerNorm(features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, sublayer: Callable[[Tensor], Tensor]) -> Tensor:
        """Apply the residual sub-layer transform.

        Args:
            x: ``(B, S, features)`` input — the residual stream.
            sublayer: A function that takes a ``(B, S, features)`` tensor and
                returns one of the same shape. In practice we pass either a
                bound method (e.g. ``self.ff``) or a small lambda that closes
                over the right arguments (mask, encoder memory, etc.).

        Returns:
            ``(B, S, features)`` — the updated residual stream.
        """
        # Read the formula left-to-right:
        #   self.norm(x)         → normalise the input
        #   sublayer(...)        → run attention or FFN
        #   self.dropout(...)    → apply dropout to its output
        #   x + ...              → residual add
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """One encoder layer: self-attention sub-layer, then feed-forward sub-layer.

    Each sub-layer is wrapped in its own ``SublayerConnection`` (so its own
    pre-norm and residual). Two sub-layers per encoder layer, six layers
    stacked = the paper's "N=6" encoder.

    Sub-modules:
        self_attn: Multi-head self-attention. "Self" because Q, K, V all
            come from the same sequence (the encoder's running state).
        ff: Position-wise feed-forward MLP — see feedforward.py.
        sub1: Residual wrapper around ``self_attn``.
        sub2: Residual wrapper around ``ff``.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sub1 = SublayerConnection(d_model, dropout)
        self.sub2 = SublayerConnection(d_model, dropout)

    def forward(self, x: Tensor, src_mask: Tensor | None = None) -> Tensor:
        """Run one encoder layer.

        Args:
            x: ``(B, S_src, d_model)`` — current encoder state.
            src_mask: ``(B, 1, 1, S_src)`` padding mask (see masking.py).
                Same mask used for every layer.

        Returns:
            ``(B, S_src, d_model)`` — updated encoder state.
        """
        # Self-attention sub-layer. ``y`` here is the post-LayerNorm input
        # produced inside SublayerConnection; we feed it to attention as all
        # three of (query, key, value) — that's what makes it self-attention.
        x = self.sub1(x, lambda y: self.self_attn(y, y, y, src_mask))
        # Feed-forward sub-layer. ``self.ff`` already takes a single tensor
        # and returns one of the same shape, so we can pass it directly.
        return self.sub2(x, self.ff)


class DecoderLayer(nn.Module):
    """One decoder layer: masked self-attention → cross-attention → feed-forward.

    Three sub-layers (vs. two in the encoder). The middle one — cross-attention
    — is what lets the decoder peek at the encoder's output: queries come
    from the decoder's own running state, but keys and values come from the
    encoder ``memory``.

    Sub-modules:
        self_attn: Masked multi-head self-attention. The mask is the
            ``tgt_mask`` from masking.py — it kills both pad positions and
            future positions, so the decoder can't cheat.
        cross_attn: Multi-head attention with Q from the decoder side and
            K, V from the encoder memory. The mask is the encoder's
            ``src_mask`` (we mustn't attend to source padding).
        ff, sub1, sub2, sub3: Same idea as the encoder layer but with three
            residual wrappers because there are three sub-layers.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sub1 = SublayerConnection(d_model, dropout)
        self.sub2 = SublayerConnection(d_model, dropout)
        self.sub3 = SublayerConnection(d_model, dropout)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ) -> Tensor:
        """Run one decoder layer.

        Args:
            x: ``(B, S_tgt, d_model)`` — current decoder state.
            memory: ``(B, S_src, d_model)`` — final encoder output. Provides
                K and V to the cross-attention sub-layer.
            src_mask: ``(B, 1, 1, S_src)`` — used by cross-attention so the
                decoder can't attend to padding in the source.
            tgt_mask: ``(B, 1, S_tgt, S_tgt)`` — combined pad+causal mask so
                self-attention is causal.

        Returns:
            ``(B, S_tgt, d_model)`` — updated decoder state.
        """
        # 1) Masked self-attention over the decoder's own state.
        x = self.sub1(x, lambda y: self.self_attn(y, y, y, tgt_mask))
        # 2) Cross-attention: queries from the decoder, keys/values from the
        #    encoder. ``memory`` is captured in the closure once per layer.
        x = self.sub2(x, lambda y: self.cross_attn(y, memory, memory, src_mask))
        # 3) Feed-forward sub-layer.
        return self.sub3(x, self.ff)
