"""Tests for our hand-rolled LayerNorm (transformer/layernorm.py).

Two checks:
  * The math: with default gamma=1, beta=0, output rows have mean ≈ 0 and
    std ≈ 1 along the last axis (the feature axis).
  * The wiring: gamma and beta exist as registered parameters with the
    expected shape, so they show up in .parameters() and get optimized.
"""

from __future__ import annotations

import torch

from transformer.layernorm import LayerNorm


def test_layernorm_normalizes_last_dim() -> None:
    """After LN with default affine, last-dim mean ≈ 0, std ≈ 1."""
    ln = LayerNorm(features=8)
    # Random input shifted and scaled to make sure the test isn't a
    # tautology — if our LN secretly returned its input unchanged, this
    # would fail because the input mean/std are not 0/1.
    x = torch.randn(4, 6, 8) * 5.0 + 3.0
    y = ln(x)
    assert y.shape == x.shape
    assert torch.allclose(y.mean(dim=-1), torch.zeros(4, 6), atol=1e-5)
    # ``unbiased=False`` matches the implementation's choice (divide by N,
    # not N-1). Loose tolerance because of the ``+ eps`` inside the sqrt.
    assert torch.allclose(y.std(dim=-1, unbiased=False), torch.ones(4, 6), atol=1e-3)


def test_layernorm_has_affine_params() -> None:
    """gamma and beta exist as named parameters with the right shape."""
    ln = LayerNorm(features=8)
    params = dict(ln.named_parameters())
    assert "gamma" in params and "beta" in params
    assert params["gamma"].shape == (8,)
    assert params["beta"].shape == (8,)
