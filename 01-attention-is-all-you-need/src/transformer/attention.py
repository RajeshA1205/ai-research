"""Scaled dot-product attention and multi-head attention.

Paper section: §3.2 ("Attention"). This is the heart of the paper —
everything else is wiring around the function defined here.

The big idea
------------
For every query vector ``q`` (one per position in the query sequence), pick
out information from a set of (key, value) pairs by:

  1. Score each key against the query using a dot product.
  2. Convert scores to a probability distribution with softmax.
  3. Use that distribution to take a weighted average of the values.

This lets each output position dynamically pull info from the most "relevant"
input positions instead of being limited to a fixed local window like a CNN.

Scaled-dot-product attention (eq. 1):

    Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V

Why divide by ``√d_k``? When ``d_k`` is large, the dot products ``q·k`` have
variance proportional to ``d_k``. Big-magnitude pre-softmax scores push the
softmax into one-hot territory, where gradients vanish. Dividing by ``√d_k``
keeps the variance roughly constant regardless of ``d_k``.

Multi-head attention
--------------------
Instead of doing one attention with width ``d_model``, the paper does ``h``
parallel attentions each with width ``d_k = d_model / h``, and concatenates
their outputs. This lets different heads attend to different kinds of
relationships (e.g. one head might track short-range syntactic deps, another
might track long-range coreference).

The heads aren't independent operations on different inputs; they share the
same input but each head sees a different *linear projection* of it. The
projections (``W_q^(i)``, ``W_k^(i)``, ``W_v^(i)``) are learned. In code we
implement all ``h`` projections in one go using a single ``Linear(d_model,
d_model)``, then reshape the output to expose the per-head axis. This is a
standard trick — mathematically equivalent to ``h`` separate small Linears,
but a single big matmul runs much faster on GPU/MPS.
"""

from __future__ import annotations

import math

from torch import Tensor, nn


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
    dropout: nn.Dropout | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute ``softmax(QKᵀ / √d_k) · V``.

    This is a pure function — no learned parameters. It works on any
    shape-compatible Q/K/V; in this codebase we always call it with the
    per-head shapes ``(B, h, S, d_k)`` produced by ``MultiHeadAttention``.

    Args:
        q: Queries, ``(B, h, S_q, d_k)``.
        k: Keys, ``(B, h, S_k, d_k)``. ``d_k`` must match ``q``.
        v: Values, ``(B, h, S_k, d_v)``. We allow ``d_v ≠ d_k`` in principle;
            in this codebase ``d_v == d_k`` always.
        mask: Optional ``(...broadcastable..., S_q, S_k)`` bool/byte tensor.
            Positions where ``mask == 0`` (i.e. False) are treated as
            "kill this attention edge" — their scores are set to ``-inf``
            before softmax, so they get exactly 0 weight.
        dropout: Optional ``nn.Dropout`` module. Applied to the post-softmax
            attention weights (the paper does this).

    Returns:
        out: ``(B, h, S_q, d_v)`` — the attended values, one weighted sum
            per query position per head.
        attn: ``(B, h, S_q, S_k)`` — the softmax weights themselves.
            Returned mainly for inspection/visualisation; the ``out`` tensor
            is what flows downstream.
    """
    # Last-dim of q is d_k; we'll divide by sqrt of it.
    d_k = q.size(-1)

    # Compute raw attention scores. q @ k.transpose(-2, -1) gives shape
    # (B, h, S_q, S_k): for every query position, a score against every key
    # position. Dividing by sqrt(d_k) keeps the variance of the scores
    # roughly constant regardless of d_k.
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply the mask (if any) by writing -inf into killed positions. After
    # softmax, exp(-inf) == 0, so those positions contribute nothing to the
    # weighted sum. We use ``mask == 0`` so the same code works whether the
    # caller passes a bool tensor (False == 0) or an int 0/1 tensor.
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # Softmax over the *key* axis: each query position produces a probability
    # distribution over the keys. Sum along dim=-1 is 1 (modulo masking out
    # entire rows — see comment below).
    attn = scores.softmax(dim=-1)

    # NOTE: if every key for some query is masked, the softmax row is all
    # -inf → softmax produces NaN. In practice this only happens for query
    # positions that are themselves padding, and those rows are masked out
    # of the loss anyway, so the NaN never affects training. The standard
    # implementations don't bother special-casing this.

    # Optional dropout on the attention weights themselves. This randomly
    # forgets some attention edges and forces the model to spread its bets.
    if dropout is not None:
        attn = dropout(attn)

    # Weighted sum: (B, h, S_q, S_k) @ (B, h, S_k, d_v) → (B, h, S_q, d_v).
    out = attn @ v
    return out, attn


class MultiHeadAttention(nn.Module):
    """h parallel attention heads with learned Q/K/V/output projections.

    The forward signature takes three tensors (q, k, v) at the d_model level,
    not after splitting into heads — splitting is an internal detail. Pass
    the same tensor three times for self-attention; pass (decoder_state,
    encoder_memory, encoder_memory) for cross-attention.

    Learned weights (parameters):
        w_q, w_k, w_v: each ``Linear(d_model, d_model)``. These hold the
            per-head Q/K/V projections, packed into one matmul. After we
            multiply and reshape, dim ``-2`` of size ``n_heads`` lets us
            address each head; dim ``-1`` of size ``d_k`` is the per-head
            feature axis.
        w_o: ``Linear(d_model, d_model)`` — the output projection that
            combines the concatenated head outputs (eq. just below paper §3.2.2).

    Cached attribute (not a parameter):
        attn: After every forward pass, this holds the most recent
            attention map ``(B, h, S_q, S_k)`` — useful when inspecting
            what the model is doing. ``.detach()``-ed, so no autograd graph
            is held on to.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        # Same divisibility check as in ModelConfig — defensive duplication.
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        # Per-head dimension. d_k = d_v in this implementation.
        self.d_k = d_model // n_heads

        # The four learned projections. Each is a single Linear that packs
        # all h heads' projections into one (d_model × d_model) matrix.
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        # Dropout on attention weights. Created once, passed into the SDPA
        # function on every forward.
        self.dropout = nn.Dropout(dropout)

        # Inspection buffer for the last attention map. Type annotation only;
        # the actual value is set on the first forward pass.
        self.attn: Tensor | None = None

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Run multi-head attention.

        Args:
            q, k, v: ``(B, S_*, d_model)`` tensors. For self-attention, all
                three are the same tensor (and S_q == S_k); for
                cross-attention, ``q`` comes from the decoder side and
                ``k`` and ``v`` come from the encoder memory.
            mask: Broadcastable ``(B, 1, S_q, S_k)`` bool mask, as built by
                ``masking.make_pad_mask`` / ``make_tgt_mask``.

        Returns:
            ``(B, S_q, d_model)`` — same shape as ``q``.
        """
        B = q.size(0)  # batch size

        # ---- 1) Project, then split into heads ---------------------------
        # self.w_q(q) keeps the (B, S, d_model) shape. We reshape the last
        # axis from d_model → (n_heads, d_k) and then transpose so the heads
        # axis comes before the sequence axis. End shape: (B, h, S, d_k).
        q_p = self.w_q(q).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_p = self.w_k(k).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_p = self.w_v(v).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        # ---- 2) Run scaled-dot-product attention per head ----------------
        # The function operates on the last two axes (sequence and feature),
        # so the heads dim just rides along — h independent attentions
        # computed in one batched call.
        out, attn = scaled_dot_product_attention(q_p, k_p, v_p, mask, self.dropout)

        # Cache the attention map for later inspection (.detach so no graph
        # is held on to and we don't leak memory across iterations).
        self.attn = attn.detach()

        # ---- 3) Concat heads back into one big vector --------------------
        # Inverse of step 1: move the heads axis next to d_k, then merge them
        # back into a single d_model axis.
        # ``contiguous()`` is required because ``transpose`` returns a view
        # with non-contiguous strides; ``view`` only works on contiguous data.
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)

        # ---- 4) Final output projection ----------------------------------
        # Mixes the per-head outputs into a single d_model-wide representation.
        return self.w_o(out)
