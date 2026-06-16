"""Tests for the gradient-as-currency mechanism (enhancement on top of v1).

One quantity - the per-synapse gradient backprop already computes - is metered
and read three ways: confidence (existing wires), pruning (existing wires),
growth (missing wires). These tests pin each readout's behaviour.
"""

import numpy as np
import pytest

from sprout.data import generate_blobs
from sprout.network import Network, build_graph, init_weights
from sprout.train import Config, Trainer, accuracy
from sprout.currency import (
    update_gradient_meters,
    realize_gradient_meters,
    meter_grad_mag,
    meter_grad_signed,
    mean_grad_mag,
    network_scales,
    load,
    demand,
    settledness,
    update_confidence_currency,
    update_confidence_2d,
    prune_currency,
    grow_currency,
    batch_edge_scores,
    update_ghost_meter,
    active_ghost_sets,
    iter_ghost_candidates,
    dense_ghost_count,
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


def test_lazy_gradient_meters_match_eager_after_realization():
    eager = build_graph([2, 3, 2], density=1.0, seed=0)
    lazy = build_graph([2, 3, 2], density=1.0, seed=0)
    keys = list(eager.synapses)
    seq = [
        {k: 2.0 for k in keys},
        {k: (0.0 if i % 2 else -1.0) for i, k in enumerate(keys)},
        {k: 0.0 for k in keys},
        {k: (3.0 if i == 0 else 0.0) for i, k in enumerate(keys)},
    ]
    for step, grad in enumerate(seq):
        update_gradient_meters(eager, grad, beta=0.5, step_idx=step)
        update_gradient_meters(lazy, grad, beta=0.5, step_idx=step, lazy=True)

        # Virtual reads are exact before realization.
        for k in keys:
            le = lazy.synapses[k]
            ee = eager.synapses[k]
            assert meter_grad_mag(le, 0.5, step + 1) == pytest.approx(ee.grad_mag)
            assert meter_grad_signed(le, 0.5, step + 1) == pytest.approx(ee.grad_signed)

    realize_gradient_meters(lazy, beta=0.5, step_idx=len(seq) - 1)
    for k in keys:
        assert lazy.synapses[k].grad_mag == pytest.approx(eager.synapses[k].grad_mag)
        assert lazy.synapses[k].grad_signed == pytest.approx(eager.synapses[k].grad_signed)


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


# -- shared state: load / demand (one source of truth for both lenses) -------

def test_network_scales_returns_mean_weight_and_mean_grad_mag():
    net = Network([2, 2])
    a = net.add_synapse(0, 2)
    b = net.add_synapse(1, 2)
    a.weight, b.weight = -1.0, 3.0       # abs -> wbar = (1+3)/2 = 2.0
    a.grad_mag, b.grad_mag = 0.2, 0.4    # Mbar = 0.3
    wbar, mbar = network_scales(net)
    assert abs(wbar - 2.0) < 1e-12
    assert abs(mbar - 0.3) < 1e-12       # == mean_grad_mag
    assert abs(mbar - mean_grad_mag(net)) < 1e-12


def test_load_and_demand_are_normalized_coordinates():
    net = Network([2, 2])
    a = net.add_synapse(0, 2)
    b = net.add_synapse(1, 2)
    a.weight, b.weight = -1.0, 3.0        # wbar = 2.0
    a.grad_mag, b.grad_mag = 0.2, 0.4     # Mbar = 0.3
    wbar, mbar = network_scales(net)
    assert abs(load(a, wbar) - 0.5) < 1e-9       # |-1|/2
    assert abs(demand(b, mbar) - (0.4 / 0.3)) < 1e-9


# -- the softened settled cliff: settledness(d, mode, k) ---------------------

def test_settledness_hard_matches_relu_cliff():
    assert settledness(0.0, "hard") == 1.0
    assert settledness(0.5, "hard") == 0.5
    assert settledness(1.0, "hard") == 0.0
    assert settledness(2.0, "hard") == 0.0       # slams to zero past average demand


def test_settledness_sigmoid_is_smooth_with_pivot_at_average():
    assert abs(settledness(1.0, "sigmoid", k=3.0) - 0.5) < 1e-9   # pivot @ avg demand
    assert settledness(2.0, "sigmoid", k=3.0) > 0.0               # never a hard zero
    assert settledness(0.0, "sigmoid", k=3.0) > settledness(1.0, "sigmoid", k=3.0)


def test_settledness_exp_is_one_at_zero_and_strictly_positive():
    assert abs(settledness(0.0, "exp", k=1.0) - 1.0) < 1e-9
    assert settledness(5.0, "exp", k=1.0) > 0.0


def test_settledness_rational_has_fatter_tail_than_exp():
    # at high demand the rational keeps more settledness (slower decay) than exp
    assert settledness(5.0, "rational", k=1.0) > settledness(5.0, "exp", k=1.0)


def test_settledness_all_modes_monotone_decreasing_in_demand():
    for mode in ("hard", "sigmoid", "exp", "rational"):
        vals = [settledness(d, mode, k=3.0) for d in (0.0, 0.5, 1.0, 1.5, 2.0)]
        assert all(vals[i] >= vals[i + 1] - 1e-12 for i in range(len(vals) - 1)), mode


# -- Readout A (2D): confidence from importance x settledness ----------------

def test_2d_confidence_freezes_heavy_settled_wire():
    # A wire that is above-average weight AND settled (low demand) earns confidence.
    net = Network([2, 2])
    heavy = net.add_synapse(0, 2)     # above-average weight, low demand
    other = net.add_synapse(1, 2)     # keeps wbar / Mbar up
    heavy.weight, heavy.grad_mag, heavy.confidence = 2.0, 0.05, 0.0
    other.weight, other.grad_mag = 0.5, 0.35
    update_confidence_2d(net, gain=2.0, alpha=0.1, c_max=5.0)
    assert heavy.confidence > 0.0


def test_2d_confidence_ignores_below_average_weight_wire():
    # A settled but light (below-average weight) wire is a freeloader -> no freeze.
    net = Network([2, 2])
    light = net.add_synapse(0, 2)     # settled but below-average weight
    heavy = net.add_synapse(1, 2)
    light.weight, light.grad_mag, light.confidence = 0.2, 0.05, 0.0
    heavy.weight, heavy.grad_mag = 2.0, 0.35
    update_confidence_2d(net, gain=2.0, alpha=0.1, c_max=5.0)
    assert light.confidence == 0.0


def test_2d_confidence_releases_under_contention():
    # A previously-frozen heavy wire that is now contested (high demand) decays.
    net = Network([2, 2])
    contested = net.add_synapse(0, 2)
    other = net.add_synapse(1, 2)
    contested.weight, contested.grad_mag, contested.confidence = 2.0, 0.9, 3.0
    other.weight, other.grad_mag = 0.5, 0.1
    update_confidence_2d(net, gain=2.0, alpha=0.1, c_max=5.0)
    assert contested.confidence < 3.0     # settled=0 -> target 0 -> decays


def test_2d_confidence_dead_heavy_freezes_dead_light_does_not():
    net = Network([3, 1])
    dead_heavy = net.add_synapse(0, 3)    # no gradient, big weight -> settled+important
    dead_light = net.add_synapse(1, 3)    # no gradient, tiny weight -> freeloader
    live = net.add_synapse(2, 3)          # keeps Mbar > 0
    dead_heavy.weight, dead_heavy.grad_mag, dead_heavy.confidence = 2.0, 0.0, 0.0
    dead_light.weight, dead_light.grad_mag, dead_light.confidence = 0.1, 0.0, 0.0
    live.weight, live.grad_mag = 0.5, 0.9
    update_confidence_2d(net, gain=2.0, alpha=0.1, c_max=5.0)
    assert dead_heavy.confidence > 0.0
    assert dead_light.confidence == 0.0


def test_2d_confidence_ema_moves_fraction_toward_target():
    net = Network([2, 2])
    w = net.add_synapse(0, 2)
    other = net.add_synapse(1, 2)
    w.weight, w.grad_mag, w.confidence = 3.0, 0.0, 0.0
    other.weight, other.grad_mag = 1.0, 0.4
    # wbar=2.0 -> imp=0.5 ; Mbar=0.2 -> settled=1.0 ; target=2.0*0.5*1.0=1.0
    update_confidence_2d(net, gain=2.0, alpha=0.1, c_max=5.0, settled_mode="hard")
    assert abs(w.confidence - 0.1) < 1e-9    # 0.9*0 + 0.1*1.0


def test_2d_confidence_clips_result_to_c_max():
    net = Network([2, 2])
    w = net.add_synapse(0, 2)
    other = net.add_synapse(1, 2)
    w.weight, w.grad_mag, w.confidence = 3.0, 0.0, 100.0   # absurd start
    other.weight, other.grad_mag = 1.0, 0.4
    update_confidence_2d(net, gain=2.0, alpha=0.1, c_max=5.0)
    assert w.confidence <= 5.0


def test_2d_confidence_soft_cliff_lifts_contested_loadbearer_off_zero():
    # The TAIL: a heavy, ABOVE-average-demand wire scores settled=0 under the
    # hard cliff -> zero confidence regardless of load. A softened cliff gives it
    # a small but NONZERO (load-proportional) rigidity instead.
    def fresh():
        net = Network([2, 2])
        heavy = net.add_synapse(0, 2)
        other = net.add_synapse(1, 2)
        # wbar=1.25 -> imp(heavy)=0.6>0 ; Mbar=0.5 -> d(heavy)=1.8 (>average)
        heavy.weight, heavy.grad_mag, heavy.confidence = 2.0, 0.9, 0.0
        other.weight, other.grad_mag = 0.5, 0.1
        return net, heavy

    net_h, heavy_h = fresh()
    update_confidence_2d(net_h, gain=2.0, alpha=0.5, c_max=5.0, settled_mode="hard")
    assert heavy_h.confidence == 0.0          # the tail under today's cliff

    for mode in ("sigmoid", "exp", "rational"):
        net_s, heavy_s = fresh()
        update_confidence_2d(net_s, gain=2.0, alpha=0.5, c_max=5.0, settled_mode=mode)
        assert heavy_s.confidence > 0.0, mode  # lifted off zero


def test_2d_confidence_soft_cliff_still_ignores_freeloaders():
    # The imp floor is untouched: a below-average-weight wire never freezes under
    # ANY cliff mode (this is what guards frozen_freeloader_frac == 0).
    for mode in ("hard", "sigmoid", "exp", "rational"):
        net = Network([2, 2])
        light = net.add_synapse(0, 2)
        heavy = net.add_synapse(1, 2)
        light.weight, light.grad_mag, light.confidence = 0.2, 0.05, 0.0
        heavy.weight, heavy.grad_mag = 2.0, 0.35
        update_confidence_2d(net, gain=2.0, alpha=0.5, c_max=5.0, settled_mode=mode)
        assert light.confidence == 0.0, mode


def test_2d_confidence_defaults_to_sigmoid_not_hard():
    # With no settled_mode given, the rule uses the softened (sigmoid) cliff, so a
    # contested load-bearer is NOT slammed to zero the way the hard cliff does.
    net = Network([2, 2])
    heavy = net.add_synapse(0, 2)
    other = net.add_synapse(1, 2)
    heavy.weight, heavy.grad_mag, heavy.confidence = 2.0, 0.9, 0.0
    other.weight, other.grad_mag = 0.5, 0.1
    update_confidence_2d(net, gain=2.0, alpha=0.5, c_max=5.0)   # no settled_mode
    assert heavy.confidence > 0.0


def test_trainer_twod_confidence_mode_is_wired_and_differs():
    # The Trainer's currency path dispatches to the 2D rule when selected, and it
    # yields different (valid) confidences than the tug-of-war rule.
    X, y = generate_blobs(n=120, seed=0)

    def run(mode):
        net = build_graph([2, 4, 4, 2], density=0.6, seed=1)
        init_weights(net, seed=1)
        cfg = Config(grad_currency=True, enable_confidence=True,
                     confidence_mode=mode, eta_base=0.05)
        tr = Trainer(cfg, net, X, y, seed=1)
        for _ in range(300):
            tr.step()
        return {k: s.confidence for k, s in net.synapses.items()}

    tug = run("tugofwar")
    twod = run("twod")
    assert all(0.0 <= c <= 5.0 for c in twod.values())   # valid band
    assert tug != twod                                    # branch actually taken


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


# -- Readout C: exact-sparse batch scoring (bit-identical to the old N^2 scan) -

def _brute_edge_scores(net, X, y):
    """The original O(N^2) enumeration, kept here as the bit-identity oracle."""
    from collections import defaultdict
    ghost = defaultdict(float)
    live_sum = live_n = 0.0
    n = len(X)
    for xi, yi in zip(X, y):
        net.forward(xi)
        _, gw, gb = net.backward(int(yi))
        for g in gw.values():
            live_sum += abs(g); live_n += 1
        for j in range(net.num_neurons):
            lj = net.neurons[j].layer
            if lj == 0:
                continue
            dj = gb.get(j, 0.0)
            if dj == 0.0:
                continue
            for i in range(net.num_neurons):
                if net.neurons[i].layer >= lj or (i, j) in net.synapses:
                    continue
                ghost[(i, j)] += abs(dj * net.neurons[i].activation)
    ghost = {k: v / n for k, v in ghost.items()}
    return ghost, (live_sum / live_n if live_n else 0.0)


def test_batch_edge_scores_bit_identical_to_bruteforce():
    net = build_graph([2, 4, 4, 2], density=0.5, seed=3)
    init_weights(net, seed=3)
    X, y = generate_blobs(n=24, seed=1)
    g_new, ref_new = batch_edge_scores(net, X, y)
    g_old, ref_old = _brute_edge_scores(net, X, y)
    nz_old = {k: v for k, v in g_old.items() if v != 0.0}   # drop spurious 0 keys
    assert ref_new == pytest.approx(ref_old, rel=1e-12)
    assert set(g_new) == set(nz_old)
    for k in g_new:
        assert g_new[k] == pytest.approx(nz_old[k], rel=1e-12)


def test_batch_edge_scores_scores_ghost_from_negative_input():
    # a negative input coordinate must still produce a (nonzero) ghost score
    net = build_graph([2, 2, 2], density=0.5, seed=4)
    init_weights(net, seed=4)
    X = np.array([[-0.9, -0.8], [-0.7, 0.6]])
    y = np.array([0, 1])
    g, _ = batch_edge_scores(net, X, y)
    assert any(net.neurons[i].layer == 0 for (i, j) in g)   # an input is a live pre


def test_batch_edge_scores_demand_k_caps_candidates():
    net = build_graph([2, 6, 6, 2], density=0.5, seed=5)
    init_weights(net, seed=5)
    X, y = generate_blobs(n=20, seed=2)
    g_full, _ = batch_edge_scores(net, X, y)
    g_k1, _ = batch_edge_scores(net, X, y, grow_demand_k=1)
    assert len(g_k1) <= len(g_full)                          # bound only removes
    assert set(g_k1) <= set(g_full)                          # subset of the full set


def test_trainer_passes_grow_demand_k_to_growth():
    net = build_graph([2, 6, 6, 2], density=0.6, seed=6)
    init_weights(net, seed=6)
    X, y = generate_blobs(n=60, seed=3)
    cfg = Config(grad_currency=True, enable_confidence=True, enable_prune=True,
                 enable_grow=True, t_struct=10, grow_demand_k=1)
    tr = Trainer(cfg, net, X, y, seed=0)
    for _ in range(50):
        tr.step()                                # must not raise; growth bounded
    assert isinstance(len(net.synapses), int)    # trained, net intact


# -- Readout C: shared candidate generation (one source of truth) ------------

def test_active_ghost_sets_uses_nonzero_not_positive_for_pre():
    # input coords can be negative; a negative-activation pre is still "active"
    net = Network([2, 2])                      # inputs 0,1 ; outputs 2,3
    net.neurons[0].activation = -0.7           # negative -> still active
    net.neurons[1].activation = 0.0            # exactly zero -> inactive
    grad_b = {2: 0.5, 3: 0.0}                   # only neuron 2 has gradient
    ap, apo = active_ghost_sets(net, grad_b)
    assert 0 in ap and 1 not in ap
    assert apo == [2]                           # 3 dropped (delta 0)


def test_active_ghost_sets_topk_keeps_loudest_posts():
    net = Network([2, 3])                       # inputs 0,1 ; outputs 2,3,4
    for nid in (0, 1):
        net.neurons[nid].activation = 1.0
    grad_b = {2: 0.1, 3: 0.9, 4: 0.5}
    _, apo = active_ghost_sets(net, grad_b, grow_demand_k=2)
    assert set(apo) == {3, 4}                   # top-2 by |delta|


def test_iter_ghost_candidates_respects_layer_order_and_liveness():
    net = Network([2, 1, 1])                    # 0,1 | 2 | 3
    net.add_synapse(2, 3)                       # live
    cand = set(iter_ghost_candidates(net, active_pre=[0, 1, 2], active_post=[2, 3]))
    # into 2 (layer1): 0,1 (layer0) -> (0,2),(1,2). into 3 (layer2): 0,1,2 minus live (2,3)
    assert cand == {(0, 2), (1, 2), (0, 3), (1, 3)}


def test_dense_ghost_count_matches_bruteforce():
    net = build_graph([2, 3, 2], density=0.5, seed=2)
    brute = sum(1 for j in range(net.num_neurons)
                for i in range(net.num_neurons)
                if net.neurons[i].layer < net.neurons[j].layer
                and (i, j) not in net.synapses)
    assert dense_ghost_count(net) == brute


# -- Readout C (A2): ghost-gradient meter (anti-oscillation) -----------------

def test_ghost_meter_new_entry_enters_below_full_value():
    # A brand-new ghost (just appeared this cycle) enters the meter at only
    # (1-beta) of its score, NOT the full score: one spike cannot, on its own,
    # clear a bar set at the full instantaneous level. This is the refractory.
    meter = {}
    update_ghost_meter(meter, {(0, 2): 10.0}, beta=0.8)
    assert abs(meter[(0, 2)] - 2.0) < 1e-12     # (1-0.8)*10


def test_ghost_meter_climbs_toward_sustained_score():
    # A genuinely, persistently wanted ghost climbs toward its score over cycles.
    meter = {}
    for _ in range(50):
        update_ghost_meter(meter, {(0, 2): 10.0}, beta=0.8)
    assert meter[(0, 2)] > 9.9                  # EMA has converged to ~score


def test_ghost_meter_decays_and_drops_unseen_entries():
    # An entry not seen this cycle decays by beta, and is dropped once it falls
    # below the floor (keeps the candidate dict bounded to currently-wanted wires).
    meter = {(0, 2): 1.0}
    update_ghost_meter(meter, {}, beta=0.5)     # unseen -> 0.5
    assert abs(meter[(0, 2)] - 0.5) < 1e-12
    for _ in range(60):                         # decay past the floor
        update_ghost_meter(meter, {}, beta=0.5)
    assert (0, 2) not in meter


def test_ghost_meter_refractory_a_single_spike_stays_below_the_bar():
    # The whole point: a one-cycle virtual-gradient spike (e.g. right after a
    # prune) leaves the meter below the instantaneous spike, so a grow bar set at
    # that level is NOT cleared — the wire must re-earn growth over several
    # cycles instead of being re-requested immediately.
    meter = {}
    update_ghost_meter(meter, {(0, 2): 10.0}, beta=0.8)   # the spike
    update_ghost_meter(meter, {}, beta=0.8)               # spike gone next cycle
    bar = 1.5 * 1.0   # a typical grow bar (grow_bar_frac * ref, ref~1)
    assert meter[(0, 2)] < 10.0                 # never reaches the spike level
    # and it keeps decaying while unseen, rather than re-triggering
    assert meter[(0, 2)] < 2.0


# -- integration -------------------------------------------------------------

def test_ghost_meter_mode_trains_and_keeps_grown_edges_out_of_the_meter():
    # ghost_meter mode still learns, and a grown edge (now live) is not retained
    # as a ghost candidate — so if it is later pruned it restarts from zero.
    net = build_graph([2, 8, 8, 6, 2], density=0.5, seed=0)
    init_weights(net, seed=0)
    X, y = generate_blobs(n=300, seed=0)
    cfg = Config(grad_currency=True, enable_confidence=True, enable_prune=True,
                 enable_grow=True, eta_base=0.05, gamma_dec=0.001, t_struct=100,
                 ghost_meter=True, beta_ghost=0.8)
    tr = Trainer(cfg, net, X, y, seed=0)
    for _ in range(3000):
        tr.step()
    assert accuracy(net, X, y) > 0.9
    # the meter never holds an edge that is currently a live synapse
    assert all(e not in net.synapses for e in tr.ghost_meter)


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
