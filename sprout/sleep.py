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
    """Early-stopping-style "patience" on an EMA of the per-step training loss
    — read at BOTH thresholds: ``settled`` (plateau) and ``startled`` (spike).

    Fires ``settled`` once the smoothed loss has gone ``patience`` steps without a
    relative improvement of at least ``tol``, and only past ``warmup``. A single
    noisy dip cannot trigger it (it must *persist*); a genuine new minimum resets
    the patience counter.

    ``startled`` is the mirror (the third phase: wake = learn, sleep =
    consolidate, startle = hire): True once the (fast) loss EMA has stayed
    above ``slow_ema * (1 + spike_tol)`` for ``spike_patience`` consecutive
    steps past warmup — a sustained demand spike, not a one-sample transient.
    The spike baseline is a SLOW EMA (``beta/10``), deliberately NOT ``best``:
    best is a min-ratchet, so one lucky downward noise excursion poisons it
    and near-zero loss the typical EMA sits permanently "spiked" above its
    own trough (measured: 29 false alarms on a stationary task). Fast-vs-slow
    is stable at stationarity — both wander together — while a real
    transition lifts the fast EMA ~10x before the slow one moves. Callers
    that never read ``startled`` are unaffected (``update()`` is unchanged).
    """

    def __init__(self, beta: float, tol: float, patience: int, warmup: int,
                 spike_tol: float = 0.5, spike_patience: int = 50,
                 spike_floor: float = 0.0):
        self.beta = beta          # EMA smoothing (~1/beta step memory)
        self.tol = tol            # relative improvement that counts as progress
        self.patience = patience  # steps without progress before "settled"
        self.warmup = warmup      # no "settled"/"startled" verdict before this
        self.spike_tol = spike_tol            # spike = fast above slow*(1+this)
        self.spike_patience = spike_patience  # sustained steps before the alarm
        # The ABSOLUTE arm of the trigger: relative rise alone misfires on
        # mid-training convergence waves and post-burst prune bumps — huge
        # vs a tiny settled EMA (up to ~20x) yet absolutely small (~0.1-0.17).
        # Real transitions push the EMA to ~0.4-3.0. The caller passes
        # ~ln(n_classes)/2 (half of chance-level CE): "actually in trouble".
        self.spike_floor = spike_floor
        self.slow_beta = beta / 10.0          # the spike BASELINE's memory
        self.loss_ema: float | None = None
        self.loss_slow: float | None = None   # "recent typical level"
        self.best = float("inf")
        self.since_improve = 0
        self.since_spike = 0
        self._last_step = -1

    def update(self, loss: float, step: int) -> bool:
        """Feed one step's loss; return True iff the net is settled *now*."""
        if self.loss_ema is None:
            self.loss_ema = loss                      # seed (no zero transient)
            self.loss_slow = loss
        else:
            self.loss_ema = (1.0 - self.beta) * self.loss_ema + self.beta * loss
            self.loss_slow = ((1.0 - self.slow_beta) * self.loss_slow
                              + self.slow_beta * loss)

        if self.loss_ema < self.best * (1.0 - self.tol):
            self.best = self.loss_ema
            self.since_improve = 0
        else:
            self.since_improve += 1

        if self.loss_ema > self.loss_slow * (1.0 + self.spike_tol):
            self.since_spike += 1                     # the spike persists
        else:
            self.since_spike = 0                      # transient: re-arm

        self._last_step = step
        return step >= self.warmup and self.since_improve >= self.patience

    @property
    def startled(self) -> bool:
        """True iff the loss spike is sustained, past warmup, AND the loss is
        absolutely high enough to mean real trouble (the spike floor).

        A property over live counters (not a cached flag), so a ``reset()`` in
        the same step — e.g. a sleep burst that just fired — immediately
        disarms it; sleep and startle can never both fire on one tick.
        """
        return (self._last_step >= self.warmup
                and self.since_spike >= self.spike_patience
                and self.loss_ema is not None
                and self.loss_ema > self.spike_floor)

    def reset(self) -> None:
        """After a structural event (sleep burst OR startle pass): require
        fresh evidence before the next.

        Keeps the smoothed loss but re-baselines ``best`` AND the slow EMA to
        it, so (1) a burst that perturbs the net must re-settle before the
        next burst — churn structurally impossible; (2) after a startle, a
        *stalled*-high loss cannot re-fire the alarm (a stall is a plateau:
        sleep's job), only further deterioration can; and (3) the
        post-transition descent now registers as improvement against the
        rebased best, so sleep fires at the new task's TRUE plateau instead of
        falsely settling mid-descent (the EMA never beating the old task's
        best).
        """
        self.since_improve = 0
        self.since_spike = 0
        self.best = self.loss_ema if self.loss_ema is not None else float("inf")
        self.loss_slow = self.loss_ema


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
