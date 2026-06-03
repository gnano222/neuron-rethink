"""Sleep consolidation: detect when the network has *settled* so pruning can run
aggressively without churn.

Measurements (docs/superpowers/specs/2026-06-03-sleep-consolidation-design.md)
showed that one-shot pruning of a *converged* net is ~lossless where the same
sparsity reached by continuous online churn destroys accuracy — so the lever is
*when* you prune, not the criterion. The only clean settledness signal is a
smoothed loss; mean confidence and the gradient meter are too noisy.

This module owns just the detector. The consolidation burst itself reuses
``currency.prune_currency`` with aggressive parameters (wired in ``Trainer``).
"""

from __future__ import annotations


class SettlednessDetector:
    """Early-stopping-style "patience" on an EMA of the per-step training loss.

    Fires ``settled`` once the smoothed loss has gone ``patience`` steps without a
    relative improvement of at least ``tol``, and only past ``warmup``. A single
    noisy dip cannot trigger it (it must *persist*); a genuine new minimum resets
    the patience counter.
    """

    def __init__(self, beta: float, tol: float, patience: int, warmup: int):
        self.beta = beta          # EMA smoothing (~1/beta step memory)
        self.tol = tol            # relative improvement that counts as progress
        self.patience = patience  # steps without progress before "settled"
        self.warmup = warmup      # no "settled" verdict before this step
        self.loss_ema: float | None = None
        self.best = float("inf")
        self.since_improve = 0

    def update(self, loss: float, step: int) -> bool:
        """Feed one step's loss; return True iff the net is settled *now*."""
        if self.loss_ema is None:
            self.loss_ema = loss                      # seed (no zero transient)
        else:
            self.loss_ema = (1.0 - self.beta) * self.loss_ema + self.beta * loss

        if self.loss_ema < self.best * (1.0 - self.tol):
            self.best = self.loss_ema
            self.since_improve = 0
        else:
            self.since_improve += 1

        return step >= self.warmup and self.since_improve >= self.patience

    def reset(self) -> None:
        """After a consolidation burst: require a fresh plateau before the next.

        Keeps the smoothed loss but forgets the prior best, so a burst that
        perturbs the net must re-settle (re-plateau) before it can fire again —
        making churn structurally impossible.
        """
        self.since_improve = 0
        self.best = self.loss_ema if self.loss_ema is not None else float("inf")
