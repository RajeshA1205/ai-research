"""Label-smoothed KL-divergence loss.

Paper section: §5.4 ("Regularization"), with ε_ls = 0.1.

What "label smoothing" means
----------------------------
Plain cross-entropy treats the target label as a one-hot vector: 1.0 on the
true class, 0.0 on every other class. The model is rewarded for putting all
its probability on the right answer.

Label smoothing softens the target: it puts ``(1 − ε)`` mass on the true
class and spreads ``ε`` evenly over all the other classes. The model is no
longer rewarded for being arbitrarily confident; getting close is good
enough. The paper notes this *hurts* perplexity (the model is forced to be
less confident, so its average predicted probability for the right answer
goes down) but *improves* BLEU and accuracy. It's a regulariser in the same
spirit as dropout.

Why KL-divergence and not cross-entropy?
----------------------------------------
With a smoothed (non-one-hot) target distribution ``p`` and predicted
distribution ``q``, you're computing

    -Σ p(y) log q(y)

which is exactly KL(p || q) up to a constant (``-Σ p(y) log p(y)``, the
target entropy, which doesn't depend on the model parameters). PyTorch
provides ``nn.KLDivLoss`` which expects log-probabilities for the input
side, hence the ``log_softmax`` on logits below.

Two extra subtleties handled here
---------------------------------
1. The pad token never gets any probability mass. We zero out column
   ``pad_id`` of the target distribution so the model isn't encouraged to
   predict pad.
2. For target tokens that ARE padding, we zero out the entire row so they
   contribute zero loss. (If we didn't, padding positions would still pull
   the model toward "predict any non-pad token uniformly".)
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LabelSmoothingLoss(nn.Module):
    """KL-divergence against a smoothed one-hot target.

    For each non-pad target token y, the smoothed target distribution puts
        (1 − smoothing) on column y
        smoothing / (V − 2) on every other column   (V is the vocab size)
        0 on column ``pad_id`` (excluded from smoothing too)
    For pad target tokens, the entire row is zero so they don't affect loss.

    The "V − 2" denominator excludes (a) the true class and (b) the pad
    column from the smoothing pool — those two columns receive their mass
    via other rules.

    Args:
        vocab_size: Number of output classes (target vocabulary size). Must
            match the model's generator output dim.
        pad_id: Token ID reserved for padding. Both pad target rows and the
            pad column of the predicted distribution are zeroed out.
        smoothing: ``ε_ls`` from the paper. 0.1 is the standard value.

    Stored attributes:
        confidence: ``1 − smoothing``, the mass placed on the true class.
            Precomputed once.
        criterion: The underlying ``nn.KLDivLoss`` with sum reduction. We
            use sum (not mean) so the caller can normalise by token count
            in the way they prefer (e.g., mean over non-pad tokens).
    """

    def __init__(self, vocab_size: int, pad_id: int = 0, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        # Sum reduction lets the training loop divide by the number of
        # non-pad tokens, matching the Annotated Transformer's loss scaling.
        self.criterion = nn.KLDivLoss(reduction="sum")

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """Compute the smoothed loss.

        Args:
            logits: ``(B*S, V)`` raw scores. The caller is expected to flatten
                the (batch, sequence) dims down to a single row dim — this is
                what training loops typically do anyway. ``V`` must equal
                ``self.vocab_size``.
            target: ``(B*S,)`` long tensor of gold token IDs.

        Returns:
            Scalar tensor — the summed KL-divergence over the batch. Caller
            divides by the number of non-pad tokens to get a per-token mean.
        """
        # KLDivLoss expects log-probabilities as input.
        log_probs = torch.log_softmax(logits, dim=-1)

        # Build the smoothed target distribution. ``no_grad`` because none of
        # this should contribute to the model's gradient — it's a constant
        # function of ``target``.
        with torch.no_grad():
            # Start with the smoothing mass spread over (V − 2) "other" classes.
            # (V − 2 = total classes minus the true class minus pad.)
            true_dist = torch.full_like(
                log_probs, self.smoothing / (self.vocab_size - 2)
            )
            # Place ``confidence`` mass on the true class for each row.
            # ``scatter_(1, idx, val)`` writes ``val`` into ``true_dist`` at
            # column index ``idx`` for each row.
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            # Zero out the pad column — the model should never get credit
            # for predicting pad, and pad shouldn't be a smoothing destination.
            true_dist[:, self.pad_id] = 0.0
            # Zero out entire rows where the target itself is pad — those
            # rows shouldn't contribute to the loss at all.
            pad_rows = (target == self.pad_id).nonzero(as_tuple=False).squeeze(-1)
            if pad_rows.numel() > 0:
                true_dist[pad_rows] = 0.0

        return self.criterion(log_probs, true_dist)
