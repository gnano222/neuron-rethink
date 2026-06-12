"""Sleep consolidation: detect when the network has *settled* so pruning can run
aggressively without churn — plus sleep-time *recycling* of dead units.

Measurements (docs/superpowers/specs/2026-06-03-sleep-consolidation-design.md)
showed that one-shot pruning of a *converged* net is ~lossless where the same
sparsity reached by continuous online churn destroys accuracy — so the lever is
*when* you prune, not the criterion. The only clean settledness signal is a
smoothed loss; mean confidence and the gradient meter are too noisy.

This module owns the detector and the recycle primitives. The consolidation
burst itself reuses ``currency.prune_currency`` with aggressive parameters
(wired in ``Trainer``); recycling is wired into the phasic rewire pass only
(docs/superpowers/specs/2026-06-11-sleep-recycling-design.md).
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


# -- sleep-time recycling: apoptosis completes, then neurogenesis -------------
#
# A dead ReLU unit is an *absorbing* state under gradient-as-currency: with
# delta = 0 it gets no parameter updates, is never scored as a ghost pre OR
# post, and so can never re-enter the economy ("no growth wasted on corpses"
# is correct — but it makes death permanent). Recycling is the re-entry path:
# at a phasic burst, a corpse's remaining wires are removed (reclaiming the
# orphan-guard zombie) and the unit is reborn as a *blank* that fires a faint
# constant — which puts it back in ``active_pre`` so the EXISTING grow scan
# prices wires from it, over the same selective bar as any candidate. Growth
# still never targets a zero-delta unit; we resurrect first, then let the
# market decide.

# A hidden unit whose firing-rate EMA (beta ~0.05) has decayed below this has
# not fired for hundreds of consecutive wake steps — and a settledness plateau
# guarantees at least ``sleep_patience`` wake steps before any burst, so truly
# dead units sit many orders of magnitude below any rarely-firing live one.
# A module constant, not a Config knob: it separates scales, it does not tune.
DEAD_RATE_FLOOR = 1e-6


def dead_by_firing_rate(net, floor: float = DEAD_RATE_FLOOR) -> list:
    """Hidden units the (otherwise vestigial) firing-rate meter says are dead.

    Free: zero extra forward passes, unlike a batch probe — and temporally
    right, since "has not fired in hundreds of steps at a plateau" is what
    *dead* means here (a small probe risks false-positive recycling of
    rare-firing live units).
    """
    last = len(net.layers) - 1
    return [nid for L in range(1, last) for nid in net.layers[L]
            if net.neurons[nid].firing_rate < floor]


def recycle_unit(net, nid, bias: float) -> None:
    """Recycle one dead unit: clear ALL its wires, rebirth as a faint blank.

    Removing the wires first (apoptosis completes) reclaims the orphan-guard
    zombie wire and guarantees the rebirth cannot perturb the function — with
    no outgoing wires the unit has zero influence. ``bias`` (> 0, the caller
    passes ``r_target``) makes it fire a constant, so it re-enters
    ``active_pre`` and bids in the next grow scan at ``|delta_post| * bias``.
    The firing-rate meter is seeded at the new steady state so the unit is not
    re-flagged while it waits to be hired. Grace does not apply: this is not
    the prune lens valuing a live wire, it is cleanup of provably-dead
    circuitry.
    """
    for pre in list(net.incoming[nid]):
        net.remove_synapse(pre, nid)
    for post in list(net.outgoing[nid]):
        net.remove_synapse(nid, post)
    neuron = net.neurons[nid]
    neuron.bias = bias
    neuron.firing_rate = bias
