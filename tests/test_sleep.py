"""Tests for the settledness detector and recycling (sprout/sleep.py)."""

from __future__ import annotations

import numpy as np
import pytest

from sprout.network import Network
from sprout.sleep import SettlednessDetector, dead_by_firing_rate, recycle_unit


def test_not_settled_before_warmup():
    # even a perfectly flat (plateaued) loss must not settle under the warmup.
    d = SettlednessDetector(beta=0.5, tol=0.01, patience=2, warmup=100)
    assert not any(d.update(1.0, step) for step in range(50))


def test_settles_after_patience_on_plateau():
    # beta=1.0 => EMA tracks raw loss exactly, so the patience logic is tested
    # without EMA-lag confounds (real runs use a small beta to smooth noise).
    d = SettlednessDetector(beta=1.0, tol=0.01, patience=3, warmup=0)
    # a strictly improving loss never settles
    assert not d.update(1.0, 0)
    assert not d.update(0.5, 1)
    assert not d.update(0.25, 2)
    # now flat: improvements stop -> after `patience` flat steps, settled
    flags = [d.update(0.25, s) for s in range(3, 9)]
    assert flags[0] is False   # not immediately
    assert flags[-1] is True   # eventually


def test_new_improvement_resets_since_improve():
    d = SettlednessDetector(beta=1.0, tol=0.01, patience=2, warmup=0)
    d.update(1.0, 0)
    d.update(1.0, 1)
    d.update(1.0, 2)                       # plateau building
    assert d.update(0.5, 3) is False       # a big improvement resets the counter
    assert d.update(0.5, 4) is False       # rebuilding patience from scratch


def test_reset_requires_fresh_plateau():
    d = SettlednessDetector(beta=1.0, tol=0.01, patience=2, warmup=0)
    for s in range(5):
        d.update(1.0, s)
    assert d.update(1.0, 5) is True        # settled after a long plateau
    d.reset()
    assert d.update(1.0, 6) is False       # must re-plateau before the next burst


def test_first_loss_seeds_the_ema():
    # the EMA seeds on the first loss (no zero-init transient that looks like a
    # huge improvement); with beta=1.0 the EMA tracks the raw loss exactly.
    d = SettlednessDetector(beta=1.0, tol=0.01, patience=1, warmup=0)
    assert d.update(0.42, 0) is False
    assert d.loss_ema == 0.42


# -- startle: the spike side of the detector ----------------------------------
#
# The spike baseline is a SLOW EMA (beta/10), not `best`: best is a min-ratchet
# (one lucky downward noise excursion poisons it, and near zero loss the
# typical EMA sits permanently "spiked" above its own trough — the smoke test
# caught exactly that storm: 29 false alarms on a stationary task). Fast-vs-
# slow is stable at stationarity and a real transition lifts fast ~10x before
# slow moves. beta=1.0 in these tests => fast EMA = raw loss, slow beta = 0.1.

def test_no_startle_during_monotone_descent():
    # while improving, fast <= slow (the slow baseline lags ABOVE): never spiked.
    d = SettlednessDetector(beta=1.0, tol=0.01, patience=100, warmup=0,
                            spike_tol=0.5, spike_patience=3)
    for s, loss in enumerate([1.0, 0.8, 0.6, 0.4, 0.3, 0.25, 0.2]):
        d.update(loss, s)
        assert d.startled is False


def test_startled_after_sustained_spike_past_patience():
    # settle at 0.1 (both EMAs converge there), then jump 4x: startled only
    # once the spike has PERSISTED spike_patience steps (a one-sample
    # transient cannot trip it). slow (beta 0.1) barely moves over 3 steps.
    d = SettlednessDetector(beta=1.0, tol=0.01, patience=1000, warmup=0,
                            spike_tol=0.5, spike_patience=3)
    for s in range(5):
        d.update(0.1, s)                  # slow baseline ~0.1
    d.update(0.4, 5)
    assert d.startled is False            # 1 step above: not yet
    d.update(0.4, 6)
    assert d.startled is False            # 2 steps: not yet
    d.update(0.4, 7)
    assert d.startled is True             # 3 sustained steps: alarm


def test_startle_respects_warmup():
    d = SettlednessDetector(beta=1.0, tol=0.01, patience=1000, warmup=100,
                            spike_tol=0.5, spike_patience=2)
    d.update(0.1, 0)
    for s in range(1, 6):
        d.update(0.9, s)                  # huge sustained spike, but pre-warmup
    assert d.startled is False


def test_plateau_wobble_below_tol_never_startles():
    # wobble that stays under slow*(1+spike_tol) resets the spike counter.
    d = SettlednessDetector(beta=1.0, tol=0.5, patience=1000, warmup=0,
                            spike_tol=0.5, spike_patience=2)
    for s, loss in enumerate([0.10, 0.10, 0.13, 0.14, 0.12, 0.14, 0.13]):
        d.update(loss, s)                 # raw never exceeds 1.5x the slow EMA
        assert d.startled is False


def test_reset_rearms_the_startle():
    # after the startle pass fires, reset() re-baselines the slow EMA to the
    # spiked level: a stalled-high loss cannot re-fire (a stall is a plateau —
    # sleep's job), only further deterioration can.
    d = SettlednessDetector(beta=1.0, tol=0.01, patience=1000, warmup=0,
                            spike_tol=0.5, spike_patience=2)
    for s in range(4):
        d.update(0.1, s)
    d.update(0.6, 4)
    d.update(0.6, 5)
    assert d.startled is True
    d.reset()                             # the startle pass fired
    assert d.startled is False
    d.update(0.6, 6)                      # stalled at the spike level
    d.update(0.6, 7)
    assert d.startled is False
    d.update(1.2, 8)                      # escalating crisis: 2x the new base
    d.update(1.2, 9)
    assert d.startled is True


def test_spike_below_absolute_floor_never_startles():
    # measured failure mode: mid-training convergence waves and post-burst
    # prune bumps are huge RELATIVE spikes (up to ~20x a tiny settled EMA) but
    # tiny in ABSOLUTE terms (~0.1-0.17) — far below chance-level loss. The
    # floor ("the net is actually in trouble") blocks them; real transitions
    # (EMA 0.4-3.0) sail over it.
    d = SettlednessDetector(beta=1.0, tol=0.01, patience=1000, warmup=0,
                            spike_tol=0.5, spike_patience=2, spike_floor=0.35)
    for s in range(4):
        d.update(0.01, s)                 # deeply settled
    d.update(0.15, 4)                     # 15x relative spike...
    d.update(0.15, 5)
    d.update(0.15, 6)
    assert d.startled is False            # ...but absolutely tiny: a wave
    d.update(0.9, 7)                      # a REAL transition
    d.update(0.9, 8)
    assert d.startled is True


def test_reset_after_spike_fixes_false_settled_mid_descent():
    # the artifact: post-transition the EMA never beats the OLD task's best,
    # so since_improve counts through the descent and sleep fires mid-learning.
    # reset() at the startle rebases best, so the descent registers as
    # improvement and settled stays False until a TRUE plateau.
    d = SettlednessDetector(beta=1.0, tol=0.01, patience=3, warmup=0,
                            spike_tol=0.5, spike_patience=2)
    for s in range(3):
        d.update(0.05, s)                 # task A: settled-low best
    d.update(0.9, 3)                      # transition
    d.update(0.9, 4)
    assert d.startled is True
    d.reset()                             # startle pass fires here
    # descending through the new task: improvements vs the REBASED best
    settled_flags = [d.update(loss, 5 + i)
                     for i, loss in enumerate([0.7, 0.5, 0.35, 0.25, 0.18])]
    assert not any(settled_flags)         # no false settled mid-descent


# -- sleep-time recycling (apoptosis completes, then neurogenesis) ------------

def _recycle_net():
    """[2,2,2] fully connected: inputs 0,1 ; hidden 2,3 ; outputs 4,5."""
    net = Network([2, 2, 2])
    for pre in (0, 1):
        for post in (2, 3):
            net.add_synapse(pre, post, weight=0.5)
    for pre in (2, 3):
        for post in (4, 5):
            net.add_synapse(pre, post, weight=0.5)
    return net


def test_dead_by_firing_rate_flags_collapsed_meter_hidden_only():
    # only HIDDEN units are recyclable: a collapsed input/output meter is not a
    # corpse (inputs hold raw data; outputs are softmax and always have delta).
    net = _recycle_net()
    for n in net.neurons:
        n.firing_rate = 0.15
    net.neurons[2].firing_rate = 0.0      # hidden corpse
    net.neurons[0].firing_rate = 0.0      # input: ignored
    net.neurons[4].firing_rate = 0.0      # output: ignored
    assert dead_by_firing_rate(net) == [2]


def test_dead_by_firing_rate_keeps_quiet_but_alive_units():
    # a unit that fired recently has r many orders of magnitude above the floor;
    # only a meter that has decayed for hundreds of silent steps qualifies.
    net = _recycle_net()
    for n in net.neurons:
        n.firing_rate = 0.15
    net.neurons[3].firing_rate = 1e-3     # quiet, but alive
    assert dead_by_firing_rate(net) == []


def test_recycle_unit_clears_all_wires_and_rebirths_as_blank():
    # apoptosis completes: ALL wires go (incoming zombie + outgoing), and the
    # unit is reborn as a blank firing at `bias` with its meter seeded there.
    net = _recycle_net()
    recycle_unit(net, 2, bias=0.15)
    assert net.incoming[2] == [] and net.outgoing[2] == []
    assert all(2 not in edge for edge in net.synapses)
    assert net.neurons[2].bias == pytest.approx(0.15)
    assert net.neurons[2].firing_rate == pytest.approx(0.15)


def test_recycled_blank_fires_and_reenters_the_grow_market():
    # the whole point: a blank fires at its bias, so it is in active_pre and the
    # EXISTING grow scan prices ghost wires from it — no new growth machinery.
    from sprout.currency import active_ghost_sets
    net = _recycle_net()
    recycle_unit(net, 2, bias=0.15)
    net.forward(np.array([0.4, -0.2]))
    assert net.neurons[2].activation == pytest.approx(0.15)
    _, _, grad_b = net.backward(0)
    active_pre, _ = active_ghost_sets(net, grad_b)
    assert 2 in active_pre
