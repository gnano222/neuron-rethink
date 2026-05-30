"""Tests for the gradient-as-currency mechanism (enhancement on top of v1).

One quantity - the per-synapse gradient backprop already computes - is metered
and read three ways: confidence (existing wires), pruning (existing wires),
growth (missing wires). These tests pin each readout's behaviour.
"""

import numpy as np

from sprout.data import generate_blobs
from sprout.network import Network, build_graph, init_weights
from sprout.train import Config, Trainer, accuracy
from sprout.currency import (
    update_gradient_meters,
    mean_grad_mag,
    update_confidence_currency,
    prune_currency,
    grow_currency,
    batch_edge_scores,
)


# -- §1 the currency: gradient meters ---------------------------------------

def test_gradient_meters_are_emas_of_magnitude_and_sign():
    net = build_graph([2, 3, 2], density=1.0, seed=0)
    init_weights(net, seed=0)
    keys = list(net.synapses)

    update_gradient_meters(net, {k: 2.0 for k in keys}, beta=0.5)
    for k in keys:
        assert abs(net.synapses[k].grad_mag - 1.0) < 1e-12      # .5*0 + .5*|2|
        assert abs(net.synapses[k].grad_signed - 1.0) < 1e-12   # .5*0 + .5*2

    update_gradient_meters(net, {k: -2.0 for k in keys}, beta=0.5)
    for k in keys:
        assert abs(net.synapses[k].grad_mag - 1.5) < 1e-12       # .5*1 + .5*2
        assert abs(net.synapses[k].grad_signed - (-0.5)) < 1e-12  # .5*1 + .5*(-2)


def test_mean_grad_mag():
    net = Network([2, 2])
    a = net.add_synapse(0, 2)
    b = net.add_synapse(1, 2)
    a.grad_mag, b.grad_mag = 0.2, 0.4
    assert abs(mean_grad_mag(net) - 0.3) < 1e-12


# -- Readout A: confidence ---------------------------------------------------

def test_confidence_rises_when_calm_and_consistent_falls_when_contested():
    net = Network([2, 2])
    calm = net.add_synapse(0, 2)      # below-average demand, consistent sign
    hot = net.add_synapse(1, 2)       # above-average demand, contested sign
    calm.grad_mag, calm.grad_signed, calm.confidence = 0.1, 0.1, 0.0
    hot.grad_mag, hot.grad_signed, hot.confidence = 0.9, 0.0, 1.0

    update_confidence_currency(net, gamma_dec=0.001, gamma_up=0.05,
                               gamma_dn=0.10, c_max=5.0, m_floor_frac=0.05)

    assert calm.confidence > 0.0   # earned confidence (calm + consistent)
    assert hot.confidence < 1.0    # lost confidence (contested hot-spot)


def test_dead_wire_gains_no_false_confidence():
    # A wire with no feedback (M ~ 0) must NOT accrue confidence just for being
    # "calm" - the consistency factor (kappa) gates it out.
    net = Network([2, 2])
    dead = net.add_synapse(0, 2)
    live = net.add_synapse(1, 2)     # keeps Mbar > 0
    dead.grad_mag, dead.grad_signed, dead.confidence = 0.0, 0.0, 0.0
    live.grad_mag, live.grad_signed = 1.0, 1.0

    update_confidence_currency(net, gamma_dec=0.001, gamma_up=0.05,
                               gamma_dn=0.10, c_max=5.0, m_floor_frac=0.05)
    assert dead.confidence == 0.0


def test_confidence_is_capped_at_c_max():
    net = Network([2, 2])
    s = net.add_synapse(0, 2)
    net.add_synapse(1, 2)
    s.grad_mag, s.grad_signed, s.confidence = 0.01, 0.01, 100.0
    update_confidence_currency(net, gamma_dec=0.001, gamma_up=0.05,
                               gamma_dn=0.10, c_max=5.0, m_floor_frac=0.05)
    assert s.confidence <= 5.0


# -- Readout B: pruning ------------------------------------------------------

def test_prune_currency_cuts_inert_protects_high_gradient_and_strong():
    net = Network([3, 2])           # layer0: 0,1,2  layer1: 3,4
    inert = net.add_synapse(0, 3)   # tiny weight, no gradient -> truly inert
    active = net.add_synapse(1, 3)  # tiny weight BUT high gradient -> protect
    strong = net.add_synapse(2, 3)  # strong weight -> protect
    for s in (inert, active, strong):
        s.age = 1000
    inert.weight, inert.grad_mag = 0.001, 0.0
    active.weight, active.grad_mag = 0.001, 1.0
    strong.weight, strong.grad_mag = 1.0, 1.0

    pruned = prune_currency(net, t_grace=10, max_prune=5,
                            prune_u_floor=0.5, lam=1.0)

    assert (0, 3) in pruned          # inert removed
    assert (1, 3) not in pruned      # small but actively-wanted: protected
    assert (2, 3) not in pruned      # strong weight: protected


def test_prune_currency_respects_grace_and_orphan_guard():
    net = Network([2, 3])              # layer0: 0,1   layer1: 2,3,4
    only = net.add_synapse(0, 2)       # neuron 2's ONLY input -> orphan guard
    only.weight, only.grad_mag, only.age = 0.0, 0.0, 1000   # inert + old
    old3 = net.add_synapse(0, 3)       # neuron 3: a strong input ...
    old3.weight, old3.grad_mag, old3.age = 1.0, 1.0, 1000
    young3 = net.add_synapse(1, 3)     # ... and a young inert one -> grace
    young3.weight, young3.grad_mag, young3.age = 0.0, 0.0, 5

    pruned = prune_currency(net, t_grace=10, max_prune=5,
                            prune_u_floor=0.5, lam=1.0)

    assert (0, 2) not in pruned        # last input to neuron 2: orphan guard
    assert (1, 3) not in pruned        # young: grace protects it


# -- Readout C: growth (RigL-style, virtual gradient) ------------------------

def test_grow_currency_grows_highest_virtual_gradient_above_bar():
    net = Network([2, 2])
    net.add_synapse(0, 2)                       # existing; (1,2),(0,3),(1,3) ghost
    ghost = {(1, 2): 5.0, (0, 3): 0.1}
    grown = grow_currency(net, ghost, ref=1.0, max_grow=2, grow_bar_frac=1.5)
    assert (1, 2) in grown                      # 5.0 > bar(1.5)
    assert (0, 3) not in grown                  # 0.1 < bar
    assert net.synapses[(1, 2)].weight == 0.0   # born at weight 0 (no disruption)


def test_batch_scores_are_zero_for_ghosts_into_dead_neuron():
    # 2-D input (matches blobs); a 2nd hidden layer so a hidden neuron has
    # unconnected candidates from layer 0 (skip) and layer 1.
    net = build_graph([2, 4, 4, 2], density=0.5, seed=1)
    init_weights(net, seed=1)
    dead = net.layers[2][0]                     # a hidden neuron in layer 2
    net.neurons[dead].bias = -1e6              # force it permanently silent

    X, y = generate_blobs(n=32, seed=0)
    ghost, ref = batch_edge_scores(net, X, y)

    # the dead neuron receives no gradient, so no wire into it is ever scored as
    # worth growing - it never appears as a grow candidate at all (the system
    # stops wasting growth on corpses for free).
    assert all(j != dead for (i, j) in ghost)
    assert ghost                                # live neurons DO get candidates
    assert ref > 0.0                            # and live edges still carry signal


# -- integration -------------------------------------------------------------

def test_currency_confidence_learns_blobs():
    net = build_graph([2, 8, 8, 6, 2], density=0.5, seed=0)
    init_weights(net, seed=0)
    X, y = generate_blobs(n=400, seed=0)
    cfg = Config(grad_currency=True, enable_confidence=True, eta_base=0.05)
    tr = Trainer(cfg, net, X, y, seed=0)
    for _ in range(3000):
        tr.step()
    assert accuracy(net, X, y) > 0.9


def test_currency_full_stack_learns_and_consolidates():
    net = build_graph([2, 8, 8, 6, 2], density=0.5, seed=0)
    init_weights(net, seed=0)
    X, y = generate_blobs(n=300, seed=0)
    cfg = Config(grad_currency=True, enable_confidence=True, enable_prune=True,
                 enable_grow=True, eta_base=0.05, gamma_dec=0.001, t_struct=100)
    tr = Trainer(cfg, net, X, y, seed=0)
    for _ in range(3000):
        tr.step()
    assert accuracy(net, X, y) > 0.9
    confs = [s.confidence for s in net.synapses.values()]
    assert max(confs) > 0.1                     # confidence actually consolidates
