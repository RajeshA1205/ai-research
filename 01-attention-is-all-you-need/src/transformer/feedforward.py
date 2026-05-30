"""Position-wise feed-forward network.

Paper section: §3.3 ("Position-wise Feed-Forward Networks").

What this is
------------
Each encoder and decoder layer has, after its attention sub-layer, a tiny
two-layer MLP applied to every position **independently**. The "position-wise"
adjective is paper-speak for "applied separately to each token vector,
without mixing across positions" — that mixing already happened in the
attention sub-layer; the FFN is purely a per-position transformation.

The math (eq. 2 in the paper):

    FFN(x) = max(0, x W₁ + b₁) W₂ + b₂

i.e. Linear → ReLU → Linear, with the inner width ``d_ff`` typically about
4× the model dimension (``d_model``=512, ``d_ff``=2048 in the paper). The
expand-then-contract pattern gives the model "room to think" in a higher-
dimensional space before projecting back to the residual-stream width.

Why is it called "position-wise" if we just apply Linear?
---------------------------------------------------------
``nn.Linear`` already operates over the last axis of its input. When you
feed it a tensor of shape ``(B, S, d_model)`` it applies the same affine
transform to every (batch, sequence) position. So a vanilla Linear layer
*is* a position-wise transform — no special looping needed.

We do add a Dropout between the ReLU and the second Linear, which the paper
also does (along with the residual+dropout that wraps the whole sub-layer).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PositionwiseFeedForward(nn.Module):
    """Two linear layers with a ReLU and dropout between them.

    Args:
        d_model: Width of the residual stream (input + output dim).
        d_ff: Width of the hidden layer. Paper uses 4× ``d_model``.
        dropout: Probability of dropping a unit between ReLU and the second
            Linear. Paper uses 0.1.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        # First projection: expand from d_model → d_ff.
        # ``nn.Linear`` includes a bias by default, which matches the b₁ in eq. 2.
        self.w_1 = nn.Linear(d_model, d_ff)
        # Second projection: contract back from d_ff → d_model so the output
        # can be added to the residual stream.
        self.w_2 = nn.Linear(d_ff, d_model)
        # Applied to the post-ReLU activations (between the two Linears).
        # Note: this is dropout *inside* the FFN sub-layer; there's another,
        # outer dropout in the SublayerConnection wrapper (see layers.py).
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Apply FFN per-position.

        Args:
            x: ``(B, S, d_model)`` activations.

        Returns:
            ``(B, S, d_model)`` — same shape; values transformed by the MLP.
        """
        # Compose the layers right-to-left, exactly matching the formula:
        #   1. self.w_1(x)         → (B, S, d_ff)
        #   2. torch.relu(...)     → (B, S, d_ff), negatives clamped to 0
        #   3. self.dropout(...)   → (B, S, d_ff), some units randomly zeroed
        #   4. self.w_2(...)       → (B, S, d_model)
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))
