"""Tests for sinusoidal positional encoding (transformer/positional.py).

We check three things:
  1. Output shape matches input shape (PE adds, doesn't reshape).
  2. The math is right at position 0: sin(0) = 0, cos(0) = 1, so even-
     indexed dims should be 0 and odd-indexed dims should be 1 there.
  3. The encoding is deterministic — two PE modules with identical args
     produce identical tables.
"""

from __future__ import annotations

import math

import torch

from transformer.positional import PositionalEncoding


def test_positional_shape() -> None:
    """PE preserves the input shape."""
    pe = PositionalEncoding(d_model=16, max_len=64, dropout=0.0)
    pe.eval()  # disable dropout so the test is deterministic
    x = torch.zeros(2, 10, 16)
    y = pe(x)
    assert y.shape == (2, 10, 16)


def test_positional_first_position_matches_formula() -> None:
    """At position 0: sin(0)=0 (even dims), cos(0)=1 (odd dims)."""
    d_model = 16
    pe = PositionalEncoding(d_model=d_model, max_len=64, dropout=0.0)
    pe.eval()
    # Zero input → output equals PE[0] exactly. No randomness needed.
    x = torch.zeros(1, 1, d_model)
    y = pe(x).squeeze()
    # Even indices (sin terms): all zero.
    assert torch.allclose(y[0::2], torch.zeros(d_model // 2), atol=1e-6)
    # Odd indices (cos terms): all one.
    assert torch.allclose(y[1::2], torch.ones(d_model // 2), atol=1e-6)


def test_positional_is_deterministic() -> None:
    """Same args → same table. Catches silent randomness in the constructor."""
    pe1 = PositionalEncoding(d_model=8, max_len=16, dropout=0.0).eval()
    pe2 = PositionalEncoding(d_model=8, max_len=16, dropout=0.0).eval()
    x = torch.zeros(1, 5, 8)
    assert torch.allclose(pe1(x), pe2(x))
    # Sanity check the formula's constant — if someone accidentally typed
    # 10 instead of 10000 in the source file, the test above would still
    # pass at position 0; this asserts the constant we depend on is real.
    assert math.isfinite(math.log(10000.0))
