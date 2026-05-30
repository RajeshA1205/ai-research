"""Tests for scaled-dot-product and multi-head attention (transformer/attention.py).

What we verify:
  * Output shape contracts (the easy bugs to introduce when reshaping for
    multi-head).
  * Softmax produces a valid distribution (sums to 1 along the key axis).
  * Masking actually zeroes the attention weights at masked positions.
  * MultiHeadAttention plumbs everything together so the input/output
    shapes at the d_model level are preserved.
"""

from __future__ import annotations

import torch

from transformer.attention import MultiHeadAttention, scaled_dot_product_attention


def test_sdpa_output_shape() -> None:
    """SDPA returns (out, attn) with the expected shapes; weights sum to 1."""
    B, h, S, d_k = 2, 4, 7, 8
    q = torch.randn(B, h, S, d_k)
    k = torch.randn(B, h, S, d_k)
    v = torch.randn(B, h, S, d_k)
    out, attn = scaled_dot_product_attention(q, k, v)
    assert out.shape == (B, h, S, d_k)
    assert attn.shape == (B, h, S, S)
    # Softmax over the key axis: each row of weights should sum to 1.
    assert torch.allclose(attn.sum(-1), torch.ones(B, h, S), atol=1e-5)


def test_sdpa_mask_zeros_attention() -> None:
    """Positions with mask==0 must receive exactly zero attention weight."""
    B, h, S, d_k = 1, 1, 4, 4
    q = torch.randn(B, h, S, d_k)
    k = torch.randn(B, h, S, d_k)
    v = torch.randn(B, h, S, d_k)
    # Hand-crafted mask: query 0 only sees keys 0 and 1.
    mask = torch.tensor([[[[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]]])
    _, attn = scaled_dot_product_attention(q, k, v, mask=mask)
    # The masked positions must be exactly zero, not just small.
    assert attn[0, 0, 0, 2].item() == 0.0
    assert attn[0, 0, 0, 3].item() == 0.0


def test_multihead_attention_output_shape() -> None:
    """MHA preserves the (B, S, d_model) shape end-to-end."""
    B, S, d_model, h = 2, 5, 32, 4
    mha = MultiHeadAttention(d_model, h, dropout=0.0)
    x = torch.randn(B, S, d_model)
    # Self-attention pattern: same tensor for q, k, v.
    y = mha(x, x, x)
    assert y.shape == (B, S, d_model)
