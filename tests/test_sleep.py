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
