"""Tests for masking helpers (transformer/masking.py).

Padding masks, causal masks, and the combined target mask used by the
decoder's masked self-attention. We test:

  * **Shape**: every mask broadcasts cleanly with attention scores
    (B, h, S_q, S_k). This is easy to get wrong (off-by-one in unsqueezes)
    and breaks attention silently — the mask just doesn't apply where you
    expect.
  * **Values**: pad mask zeroes pad columns; causal mask is lower-triangular;
    combined mask zeroes pad columns of an otherwise-causal triangle.
"""

from __future__ import annotations

import torch

from transformer.masking import make_causal_mask, make_pad_mask, make_tgt_mask


def test_pad_mask_shape_and_values() -> None:
    """Pad positions in the input become False; everything else is True."""
    seq = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
    mask = make_pad_mask(seq, pad_id=0)
    # (B=2, head-broadcast=1, query-broadcast=1, S=5)
    assert mask.shape == (2, 1, 1, 5)
    # First example: tokens 1,2,3 are content; positions 3,4 are padding.
    assert mask[0, 0, 0].tolist() == [1, 1, 1, 0, 0]
    assert mask[1, 0, 0].tolist() == [1, 1, 0, 0, 0]


def test_causal_mask_is_lower_triangular() -> None:
    """Position i can attend to positions 0..i, but not i+1..end."""
    m = make_causal_mask(4)
    assert m.shape == (1, 1, 4, 4)
    # ``torch.tril(ones)`` is the canonical reference for "lower triangular".
    expected = torch.tril(torch.ones(4, 4))
    # Our mask is bool; cast to float32 to compare against the reference.
    assert torch.equal(m.squeeze().to(torch.float32), expected)


def test_tgt_mask_combines_pad_and_causal() -> None:
    """Combined mask: pad column zeroed AND upper-triangle zeroed."""
    tgt = torch.tensor([[1, 2, 3, 0]])  # last position is pad
    mask = make_tgt_mask(tgt, pad_id=0)
    assert mask.shape == (1, 1, 4, 4)
    # Column 3 corresponds to the pad token. After combining, no query
    # should be allowed to attend to it — column sum = 0.
    assert mask[0, 0, :, 3].sum().item() == 0
