"""Noam learning-rate schedule.

Paper section: §5.3 ("Optimizer"). Used together with Adam (β1=0.9, β2=0.98,
ε=1e-9) — those are set in the training loop, not here.

The schedule (paper, eq. just below §5.3):

    lr(step) = d_model^(-0.5) * min( step^(-0.5), step * warmup^(-1.5) )

What the formula does
---------------------
* For ``step ≤ warmup``: linear warmup from 0 up to a peak. The
  ``step * warmup^(-1.5)`` term grows linearly with step, so the lr ramps
  up smoothly from zero. This avoids the very-early-training instability
  that hits big Transformers when you start at the full learning rate.
* For ``step > warmup``: inverse-square-root decay. The ``step^(-0.5)``
  term takes over and slowly shrinks the lr.
* The two terms cross at ``step == warmup`` — that's the peak value of
  the schedule.
* The leading ``d_model^(-0.5)`` factor scales the whole curve down for
  bigger models. (Larger ``d_model`` → smaller updates needed.)

Why this matters less now
-------------------------
The paper used **post-norm** Transformers, which don't train without warmup —
the gradients explode in the first few steps. With **pre-norm** (this
codebase) the model trains fine with a plain constant or cosine schedule.
We implement Noam anyway for fidelity to the paper and so you can
A/B-compare.

Why we wrote this ourselves instead of using ``torch.optim.lr_scheduler``
-------------------------------------------------------------------------
PyTorch's ``LambdaLR`` could express the same thing in a few lines. But
the paper formula is self-contained and ours is short — implementing it
explicitly makes the schedule audit-able and removes one layer of
indirection from a "from scratch" project.
"""

from __future__ import annotations

from torch.optim import Optimizer


class NoamScheduler:
    """Manual scheduler. Call ``step()`` after every optimizer step.

    Note this is **not** a subclass of ``torch.optim.lr_scheduler._LRScheduler``;
    we deliberately keep it standalone to make the formula visible and avoid
    PyTorch's somewhat baroque scheduler base class. The trade-off is that
    things like ``.state_dict()`` round-tripping for resume-from-checkpoint
    aren't implemented — extend this if you need them.

    Args:
        optimizer: The Adam (or other) optimizer to drive. We mutate its
            ``param_groups[i]['lr']`` directly on every step.
        d_model: Model width. Larger d_model → smaller schedule. Used
            literally as the ``d_model^(-0.5)`` factor in the formula.
        warmup_steps: How many steps of linear warmup before the inverse-
            sqrt decay takes over. Paper uses 4000 for the base model. For
            the toy copy task we use 400 because we only train for ~500 steps.
        factor: An overall multiplier applied on top of the formula. Useful
            if you want to scale the whole curve up/down without messing
            with d_model. Default 1.0 reproduces the paper exactly.

    Internal state:
        _step: The number of times ``step()`` has been called. Indexed from
            0 before the first call, so the first lr we set corresponds to
            step=1 (avoids the step=0 division below).
        _last_lr: The most recently applied learning rate, exposed via the
            ``last_lr`` property for convenient logging.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        d_model: int,
        warmup_steps: int = 4000,
        factor: float = 1.0,
    ) -> None:
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self._step = 0
        self._last_lr = 0.0

    def rate(self, step: int | None = None) -> float:
        """Compute the lr for a given step (or the current internal step).

        Exposed as a public method so callers can probe the curve without
        actually applying it (useful for plotting / debugging).
        """
        if step is None:
            step = self._step
        # Guard against step < 1: ``step ** -0.5`` blows up at step=0.
        # We treat the very first call as step=1.
        if step < 1:
            step = 1
        return self.factor * (self.d_model ** -0.5) * min(
            step ** -0.5, step * self.warmup_steps ** -1.5
        )

    def step(self) -> None:
        """Advance one step and write the new lr into every param group.

        Convention: call this AFTER ``optimizer.step()`` in the training
        loop, the same way you would with PyTorch's built-in schedulers.
        """
        self._step += 1
        lr = self.rate()
        # Most users have a single param group, but be permissive: if the
        # optimizer was set up with multiple groups (e.g. different lr
        # multipliers for different submodules) we still update them all.
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        self._last_lr = lr

    @property
    def last_lr(self) -> float:
        """Most recently applied lr — handy for logging in the train loop."""
        return self._last_lr
