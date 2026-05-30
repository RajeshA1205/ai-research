"""Hand-rolled layer normalization.

We deliberately do **not** use ``torch.nn.LayerNorm``. The whole point of this
project is to build the Transformer's components from scratch, and LayerNorm
is one of those components — Ba, Kiros & Hinton (2016), used at every
sub-layer in the paper. Implementing it ourselves takes about ten lines and
makes the math fully visible.

What LayerNorm does
-------------------
For each "row" of activations (the last axis of a tensor, typically the
``d_model``-wide residual stream at one position of one example in the
batch), it:

  1. Subtracts the row's mean → centers the activations.
  2. Divides by the row's standard deviation → scales them to unit variance.
  3. Applies a learnable affine transform ``γ ⊙ x + β`` → lets the model
     undo the normalization in case it's harmful (rarely is, but the
     parameters are essentially free).

Why this stabilises training: each sub-layer's output now lives in a
predictable scale (mean 0, std 1 before affine), so deep stacks don't drift
into either zero-activation collapse or exploding magnitudes.

Why **layer** norm, not **batch** norm: in NLP each token is its own little
world; we don't want statistics mixed across positions, and certainly not
across examples in the batch. LayerNorm normalises along the feature axis,
which is independent of batch size and sequence length.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LayerNorm(nn.Module):
    """Normalize over the last dimension with learnable affine parameters.

        y = gamma * (x - mean) / sqrt(var + eps) + beta

    where mean and var are computed along the last axis of ``x``.

    Args:
        features: Length of the feature axis (the size of the last dim of
            the inputs we'll receive). For a Transformer this is ``d_model``.
            We need to know it at construction time so we can size ``gamma``
            and ``beta`` correctly.
        eps: Small constant added inside the square root for numerical
            stability — without it, a row that happens to have zero variance
            would divide by zero. ``1e-6`` is the value used by the
            Annotated Transformer.

    Learnable parameters:
        gamma: ``(features,)`` — scale. Initialised to all-ones, so the
            initial transform is exactly the standardise-and-pass-through
            identity.
        beta: ``(features,)`` — shift. Initialised to all-zeros, same
            reason.
    """

    def __init__(self, features: int, eps: float = 1e-6) -> None:
        super().__init__()
        # Wrapping in nn.Parameter registers these as model parameters, so
        # they show up in .parameters(), get gradient updates, and move with
        # .to(device).
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Normalize ``x`` along its last axis.

        Args:
            x: ``(..., features)`` tensor. Any leading shape is fine — only
                the last axis is normalized. In the Transformer the typical
                shape is ``(B, S, d_model)``.

        Returns:
            Same shape as ``x``, with last-dim mean ≈ 0 and last-dim std ≈ 1
            before the gamma/beta affine.
        """
        # ``keepdim=True`` keeps the size-1 normalized axis around so that
        # the subtraction below broadcasts cleanly: (..., features) - (..., 1).
        mean = x.mean(dim=-1, keepdim=True)
        # ``unbiased=False`` matches the formula in the original LayerNorm paper:
        # divide by N rather than N-1. PyTorch's nn.LayerNorm uses the same
        # convention. Without this flag, our LayerNorm would silently produce
        # different numbers from the standard one, which is a confusing bug.
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # Standardise, then apply the learnable affine.
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
