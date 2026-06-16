"""Tests for the pure metric functions.

Every expected value here is hand-computed or built from a deterministic tiny
network / synthetic event log, so a regression points at a real arithmetic bug.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sprout.network import Network, build_graph, init_weights
from sprout.data import generate_blobs
from evals import metrics


# -- helpers -----------------------------------------------------------------

def _net_2_2_2():
    """[2,2,2]: inputs 0,1 ; hidden 2,3 ; outputs 4,5. Fully connected."""
    net = Network([2, 2, 2])
    for pre in (0, 1):
        for post in (2, 3):
            net.add_synapse(pre, post, weight=0.5)
    for pre in (2, 3):
        for post in (4, 5):
            net.add_synapse(pre, post, weight=0.5)
    return net


# -- prediction performance --------------------------------------------------

def test_test_loss_matches_negative_log_prob_of_true_class():
    net = _net_2_2_2()
    x = np.array([0.3, -0.7])
    probs = net.forward(x)            # run forward to know the probabilities
    for y in (0, 1):
        expected = -math.log(probs[y] + 1e-12)
        assert metrics.test_loss(net, [x], [y]) == pytest.approx(expected, rel=1e-9)


def test_test_loss_averages_over_samples():
    net = _net_2_2_2()
    X = [np.array([0.2, 0.4]), np.array([-0.1, 0.9])]
    y = [0, 1]
    per = [metrics.test_loss(net, [X[i]], [y[i]]) for i in range(2)]
    assert metrics.test_loss(net, X, y) == pytest.approx(sum(per) / 2, rel=1e-9)


# -- effective capacity ------------------------------------------------------

def test_dead_unit_count_flags_permanently_silent_neuron():
    net = _net_2_2_2()
    net.neurons[2].bias = -1e6          # neuron 2 can never fire (always z<0)
    net.neurons[3].bias = 1.0           # neuron 3 always fires
    X = [np.array([0.5, 0.5]), np.array([-0.3, 0.8]), np.array([0.9, -0.2])]
    assert metrics.dead_unit_count(net, X) == 1


def test_dead_unit_count_zero_when_all_fire():
    net = _net_2_2_2()
    net.neurons[2].bias = 1.0
    net.neurons[3].bias = 1.0
    X = [np.array([0.5, 0.5])]
    assert metrics.dead_unit_count(net, X) == 0


def test_neuron_activation_stats_average_and_dead_fraction():
    # "average neuron value": mean hidden-neuron ReLU output over the data,
    # averaged per-neuron-then-over-neurons (equal weight per neuron).
    # hidden 2 has bias -1e6 (never fires); hidden 3 has bias 0.5. Each hidden
    # neuron has incoming 0->h (w=0.5) and 1->h (w=0.5).
    net = _net_2_2_2()
    net.neurons[2].bias = -1e6
    net.neurons[3].bias = 0.5
    X = [np.array([1.0, 1.0]), np.array([0.0, 0.0])]
    # neuron 2: ReLU(<0) = 0 on both samples -> per-neuron mean 0.0
    # neuron 3: ReLU(0.5+1.0)=1.5 ; ReLU(0.5+0.0)=0.5 -> per-neuron mean 1.0
    # mean over the two hidden neurons = (0.0 + 1.0) / 2 = 0.5
    s = metrics.neuron_activation_stats(net, X)
    assert s["mean_neuron_activation"] == pytest.approx(0.5)
    assert s["dead_unit_frac"] == pytest.approx(0.5)   # neuron 2 of {2,3} dead


def test_neuron_activation_stats_no_hidden_neurons():
    net = Network([2, 2])               # no hidden layer
    s = metrics.neuron_activation_stats(net, [np.array([0.3, 0.7])])
    assert s["mean_neuron_activation"] == 0.0
    assert s["dead_unit_frac"] == 0.0
    assert s["idle_unit_frac"] == 0.0


def test_idle_unit_frac_counts_outputless_units():
    # hidden 2 fires but has NO outgoing wires: it contributes nothing to the
    # function => idle. hidden 3 fires and feeds the outputs => busy. This is
    # the metric recycling cannot game (a recycled blank fires, so it is not
    # "dead", but it stays idle until the market actually rehires it).
    net = _net_2_2_2()
    net.neurons[2].bias = 1.0
    net.neurons[3].bias = 1.0
    net.remove_synapse(2, 4)
    net.remove_synapse(2, 5)
    X = [np.array([0.5, 0.5])]
    s = metrics.neuron_activation_stats(net, X)
    assert s["dead_unit_frac"] == pytest.approx(0.0)   # both fire
    assert s["idle_unit_frac"] == pytest.approx(0.5)   # but 2 is outputless


def test_idle_unit_frac_counts_dead_units_even_when_fully_wired():
    net = _net_2_2_2()
    net.neurons[2].bias = -1e6          # dead, though all wires are live
    net.neurons[3].bias = 1.0
    X = [np.array([0.5, 0.5])]
    s = metrics.neuron_activation_stats(net, X)
    assert s["idle_unit_frac"] == pytest.approx(0.5)


# -- recycling ----------------------------------------------------------------

def test_recycle_metrics_counts_events_and_end_state_rehires():
    # units 2 and 3 were both recycled during the run (3 events, 2 distinct).
    # At the end: unit 2 fires and feeds the outputs (rehired); unit 3 fires
    # but is outputless (still a blank) => rehired frac = 1/2.
    net = _net_2_2_2()
    net.neurons[2].bias = 0.15
    net.neurons[3].bias = 0.15
    net.remove_synapse(3, 4)
    net.remove_synapse(3, 5)
    events = [
        {"step": 100, "type": "recycle", "edge": None, "neuron": 2},
        {"step": 100, "type": "recycle", "edge": None, "neuron": 3},
        {"step": 300, "type": "recycle", "edge": None, "neuron": 3},
    ]
    X = [np.array([0.5, 0.5])]
    s = metrics.recycle_metrics(events, net, X)
    assert s["n_recycle_events"] == 3
    assert s["recycled_rehired_frac"] == pytest.approx(0.5)


def test_recycle_metrics_nan_when_never_recycled():
    # variants without recycling: activity 0, rehired frac undefined (NaN).
    net = _net_2_2_2()
    s = metrics.recycle_metrics([], net, [np.array([0.5, 0.5])])
    assert s["n_recycle_events"] == 0
    assert math.isnan(s["recycled_rehired_frac"])


def test_recycle_metrics_are_registered():
    for k in ("idle_unit_frac", "n_recycle_events", "recycled_rehired_frac"):
        assert k in metrics.METRIC_DIRECTIONS
        assert k in metrics.METRIC_DESCRIPTIONS
    assert metrics.METRIC_DIRECTIONS["idle_unit_frac"] == "lower"


# -- startle ------------------------------------------------------------------

def test_structural_metrics_count_startle_events():
    events = [
        {"step": 100, "type": "startle", "edge": None},
        {"step": 100, "type": "grow", "edge": (0, 5)},
        {"step": 200, "type": "arousal", "edge": None},
        {"step": 200, "type": "grow", "edge": (1, 5)},
        {"step": 400, "type": "startle", "edge": None},
        {"step": 900, "type": "sleep", "edge": None},
        {"step": 900, "type": "prune", "edge": (0, 5)},
    ]
    s = metrics.structural_metrics(events)
    assert s["n_startle_events"] == 2
    assert s["n_arousal_events"] == 1
    assert s["n_grow_events"] == 2            # startles/arousal markers are not grow events


def test_startle_metric_is_registered():
    for k in ("n_startle_events", "n_arousal_events"):
        assert metrics.METRIC_DIRECTIONS[k] == "neutral"
        assert k in metrics.METRIC_DESCRIPTIONS


# -- utility -----------------------------------------------------------------

def test_synapse_utilities_combine_weight_and_demand():
    net = Network([2, 2])               # inputs 0,1 ; outputs 2,3
    net.add_synapse(0, 2, weight=2.0)
    net.add_synapse(1, 3, weight=4.0)
    demand = {(0, 2): 1.0, (1, 3): 3.0}
    # mean|w| = 3, mean demand = 2, lambda = 1
    u = metrics.synapse_utilities(net, demand, lam=1.0)
    assert u[(0, 2)] == pytest.approx(2 / 3 + 1 / 2)
    assert u[(1, 3)] == pytest.approx(4 / 3 + 3 / 2)


def test_utility_stats_freeloader_fraction():
    utilities = {("a",): 0.1, ("b",): 0.2, ("c",): 1.0, ("d",): 2.0}
    s = metrics.utility_stats(utilities, prune_u_floor=0.5)
    assert s["freeloader_frac"] == pytest.approx(0.5)   # 2 of 4 below 0.5
    assert s["mean_utility"] == pytest.approx((0.1 + 0.2 + 1.0 + 2.0) / 4)


# -- confidence calibration --------------------------------------------------

def test_confidence_calibration_perfect_correlation():
    net = Network([1, 3])               # input 0 ; outputs 1,2,3
    for i, post in enumerate((1, 2, 3)):
        syn = net.add_synapse(0, post, weight=1.0)
        syn.confidence = float(i + 1)   # 1, 2, 3
    utilities = {(0, 1): 2.0, (0, 2): 4.0, (0, 3): 6.0}  # u = 2*confidence
    cal = metrics.confidence_calibration(net, utilities)
    assert cal["conf_utility_corr"] == pytest.approx(1.0, abs=1e-9)
    assert cal["frozen_freeloader_frac"] == pytest.approx(0.0)


def test_confidence_calibration_counts_frozen_freeloaders():
    net = Network([1, 2])
    s1 = net.add_synapse(0, 1, weight=1.0); s1.confidence = 3.0   # frozen
    s2 = net.add_synapse(0, 2, weight=1.0); s2.confidence = 0.0
    utilities = {(0, 1): 0.1, (0, 2): 5.0}   # frozen one is a freeloader
    cal = metrics.confidence_calibration(net, utilities,
                                         conf_threshold=1.0, prune_u_floor=0.5)
    assert cal["frozen_freeloader_frac"] == pytest.approx(0.5)


# -- structural churn (from the event log) -----------------------------------

def test_max_grows_into_one_neuron():
    events = [
        {"step": 10, "type": "grow", "edge": (0, 5)},
        {"step": 20, "type": "grow", "edge": (1, 5)},
        {"step": 30, "type": "grow", "edge": (2, 6)},
        {"step": 40, "type": "prune", "edge": (0, 5)},
    ]
    s = metrics.structural_metrics(events)
    assert s["max_grows_into_one_neuron"] == 2   # neuron 5 grown into twice
    assert s["n_grow_events"] == 3
    assert s["n_prune_events"] == 1


# -- lifespan & oscillation (the open problem this harness must surface) ------

def test_oscillation_and_lifespan_from_grow_prune_regrow():
    # the spec's worked example: an edge grown, pruned, then regrown
    events = [
        {"step": 100, "type": "grow", "edge": (3, 7)},
        {"step": 150, "type": "prune", "edge": (3, 7)},
        {"step": 300, "type": "grow", "edge": (3, 7)},
    ]
    osc = metrics.oscillation_metrics(events)
    assert osc["max_regrow"] == 1            # grown twice => regrown once
    assert osc["oscillation_frac"] == pytest.approx(1.0)  # 1 of 1 grown edges

    lifespans = metrics.pruned_lifespans(events, initial_edges=set())
    assert lifespans == [50]                 # 150 - 100


def test_pruned_lifespan_of_initial_edge_counts_from_zero():
    events = [{"step": 80, "type": "prune", "edge": (0, 4)}]
    lifespans = metrics.pruned_lifespans(events, initial_edges={(0, 4)})
    assert lifespans == [80]                 # born at step 0


def test_oscillation_frac_zero_when_no_edge_grown_twice():
    events = [
        {"step": 10, "type": "grow", "edge": (0, 5)},
        {"step": 20, "type": "grow", "edge": (1, 6)},
    ]
    osc = metrics.oscillation_metrics(events)
    assert osc["oscillation_frac"] == pytest.approx(0.0)
    assert osc["max_regrow"] == 0


# -- training efficacy (pure over the recorded series) -----------------------

def test_steps_to_threshold():
    rec = [0, 100, 200, 300]
    acc = [0.5, 0.8, 0.92, 0.96]
    assert metrics.steps_to_threshold(rec, acc, 0.90) == 200
    assert metrics.steps_to_threshold(rec, acc, 0.95) == 300
    assert metrics.steps_to_threshold(rec, acc, 0.99) == math.inf


def test_cost_to_threshold_uses_matching_cost_series():
    costs = [10, 50, 120, 240]
    acc = [0.5, 0.8, 0.92, 0.96]
    assert metrics.cost_to_threshold(costs, acc, 0.90) == 120
    assert metrics.cost_to_threshold(costs, acc, 0.95) == 240
    assert metrics.cost_to_threshold(costs, acc, 0.99) == math.inf


def test_auc_of_accuracy_curve():
    # straight line 0 -> 1 over [0,100] has normalised area 0.5
    assert metrics.auc([0, 100], [0.0, 1.0]) == pytest.approx(0.5)
    # flat 0.9 line has normalised area 0.9
    assert metrics.auc([0, 100, 200], [0.9, 0.9, 0.9]) == pytest.approx(0.9)


def test_stability_is_std_of_tail():
    acc = [0.1, 0.2, 0.90, 0.92, 0.94]
    assert metrics.stability(acc, k=3) == pytest.approx(np.std([0.90, 0.92, 0.94]))


# -- structure: fan / density ------------------------------------------------

def test_fan_stats_fully_connected():
    net = _net_2_2_2()                  # fully connected => fan 2 each way
    s = metrics.fan_stats(net)
    assert s["mean_fan_in"] == pytest.approx(2.0)
    assert s["mean_fan_out"] == pytest.approx(2.0)
    assert s["effective_density"] == pytest.approx(1.0)   # 8 live / 8 possible


def test_effective_density_drops_when_edge_removed():
    net = _net_2_2_2()
    net.remove_synapse(0, 2)
    assert metrics.fan_stats(net)["effective_density"] == pytest.approx(7 / 8)


def test_active_edge_stats_count_forward_backward_and_gradient_edges():
    net = _net_2_2_2()
    # Hidden 2 is silent; hidden 3 fires. Output deltas are always nonzero.
    net.neurons[2].bias = -1e6
    net.neurons[3].bias = 1.0
    net.synapses[(3, 5)].weight = -0.25  # make hidden 3's backprop delta nonzero
    X = [np.array([0.5, 0.5])]
    y = [0]
    s = metrics.active_edge_stats(net, X, y)
    assert s["hidden_firing_frac"] == pytest.approx(0.5)
    # Forward-active: 4 input->hidden edges have active inputs, plus 2 edges
    # from fired hidden 3 to outputs => 6 of 8.
    assert s["fwd_active_edge_frac"] == pytest.approx(6 / 8)
    # Backward-active: output posts active for 4 hidden->output edges; hidden 3
    # has nonzero delta for its 2 input->hidden incoming edges => 6 of 8.
    assert s["bwd_active_edge_frac"] == pytest.approx(6 / 8)
    # Gradient-active needs active pre AND active post: input->hidden3 (2) plus
    # hidden3->outputs (2) => 4 of 8.
    assert s["grad_active_edge_frac"] == pytest.approx(4 / 8)


# -- quality: survivor age ---------------------------------------------------

def test_survivor_age_stats():
    net = Network([1, 3])
    for i, post in enumerate((1, 2, 3)):
        syn = net.add_synapse(0, post)
        syn.age = (i + 1) * 10           # 10, 20, 30
    s = metrics.survivor_age_stats(net)
    assert s["mean_survivor_age"] == pytest.approx(20.0)
    assert s["median_survivor_age"] == pytest.approx(20.0)


# -- capacity ----------------------------------------------------------------

def test_capacity_metrics():
    net = _net_2_2_2()
    net.neurons[2].bias = -1e6           # one dead hidden unit
    net.neurons[3].bias = 1.0
    net.synapses[(0, 2)].weight = 0.0    # one inert (zero-weight) synapse of 8
    X = [np.array([0.5, 0.5]), np.array([0.9, 0.1])]
    s = metrics.capacity_metrics(net, X, initial_count=10)
    assert s["dead_unit_count"] == 1
    assert s["inert_synapse_frac"] == pytest.approx(1 / 8)
    assert s["used_vs_allocated"] == pytest.approx(8 / 10)


# -- meter fidelity (currency sanity) ----------------------------------------

def test_meter_fidelity_perfect_correlation():
    net = Network([1, 3])
    grad = [1.0, 2.0, 3.0]
    for i, post in enumerate((1, 2, 3)):
        syn = net.add_synapse(0, post)
        syn.grad_mag = grad[i]
    demand = {(0, 1): 0.5, (0, 2): 1.0, (0, 3): 1.5}   # = 0.5 * grad_mag
    assert metrics.meter_fidelity(net, demand) == pytest.approx(1.0, abs=1e-9)


def test_meter_fidelity_nan_with_too_few_edges():
    net = Network([1, 1])
    net.add_synapse(0, 1).grad_mag = 1.0
    assert math.isnan(metrics.meter_fidelity(net, {(0, 1): 1.0}))


# -- fresh demand ------------------------------------------------------------

def test_fresh_demand_covers_all_synapses_and_is_nonnegative():
    net = _net_2_2_2()
    X = [np.array([0.3, -0.2]), np.array([0.7, 0.1])]
    y = [0, 1]
    d = metrics.fresh_demand(net, X, y)
    assert set(d) == set(net.synapses)
    assert all(v >= 0.0 for v in d.values())
    assert d == metrics.fresh_demand(net, X, y)   # deterministic / no mutation


# -- training efficacy: concept-shift recovery -------------------------------

def test_recovery_metrics():
    rec = [0, 100, 200, 300, 400, 500]
    acc = [0.5, 0.9, 0.95, 0.4, 0.7, 0.96]
    r = metrics.recovery_metrics(rec, acc, shift_start_index=3)
    assert r["pre_shift_acc"] == pytest.approx(0.95)
    assert r["recovered_acc"] == pytest.approx(0.96)
    assert r["recovery_gap"] == pytest.approx(-0.01)
    assert r["recovery_steps"] == pytest.approx(200.0)   # 500 - 300


def test_recovery_steps_inf_when_never_regained():
    rec = [0, 100, 200, 300, 400]
    acc = [0.5, 0.9, 0.95, 0.4, 0.6]     # never back to 0.95
    r = metrics.recovery_metrics(rec, acc, shift_start_index=3)
    assert r["recovery_steps"] == math.inf


# -- continual-learning (forgetting) metrics ---------------------------------

def _continual_series():
    # phase A learns A to 0.95; phase B trains only B, so A erodes to 0.40 while
    # B climbs to 0.95; phase A+B consolidates both back up.
    return {
        "phase":           ["A",  "A",  "B",  "B",  "AB", "AB"],
        "test_accuracy_A": [0.50, 0.95, 0.60, 0.40, 0.70, 0.92],
        "test_accuracy_B": [0.50, 0.50, 0.70, 0.95, 0.80, 0.90],
    }


def test_continual_metrics_forgetting_and_consolidation():
    m = metrics.continual_metrics(_continual_series())
    assert m["a_peak"] == pytest.approx(0.95)            # A at end of phase A
    assert m["b_learned"] == pytest.approx(0.95)         # B at end of phase B
    assert m["forgetting"] == pytest.approx(0.95 - 0.40)  # A's drop during B
    assert m["consolidation"] == pytest.approx(0.90)     # min(A,B) at end of A+B
    assert m["relearn_gap"] == pytest.approx(0.95 - 0.92)  # A not fully restored


def test_phase_steps_to_threshold_measures_from_phase_start():
    series = {"phase": ["A", "A", "B", "B"], "rec_step": [0, 100, 200, 300],
              "test_accuracy_B": [0.4, 0.4, 0.7, 0.95]}
    # phase B begins at step 200; B first reaches 0.90 at step 300 => 100 steps in
    assert metrics.phase_steps_to_threshold(
        series, "B", "test_accuracy_B", 0.90) == pytest.approx(100.0)
    # threshold never cleared within the phase => inf
    assert math.isinf(metrics.phase_steps_to_threshold(
        series, "B", "test_accuracy_B", 0.99))
    # a phase that never occurs => inf
    assert math.isinf(metrics.phase_steps_to_threshold(
        series, "C", "test_accuracy_B", 0.5))


def test_continual_metrics_report_per_task_learning_speed():
    # how quickly each task is learned, measured from the start of its phase.
    series = {
        "phase":           ["A",  "A",  "A",  "B",  "B",  "B"],
        "rec_step":        [0,    100,  200,  300,  400,  500],
        "test_accuracy_A": [0.50, 0.70, 0.92, 0.92, 0.60, 0.55],
        "test_accuracy_B": [0.50, 0.50, 0.50, 0.55, 0.85, 0.95],
    }
    m = metrics.continual_metrics(series)
    # task A (first task): >=0.80 and >=0.90 both first hit at step 200, phase A
    # began at step 0
    assert m["a_steps_to_80"] == pytest.approx(200.0)
    assert m["a_steps_to_90"] == pytest.approx(200.0)
    # task B (the second task): phase B began at step 300; >=0.80 at 400 (=> 100),
    # >=0.90 at 500 (=> 200)
    assert m["b_steps_to_80"] == pytest.approx(100.0)
    assert m["b_steps_to_90"] == pytest.approx(200.0)


def test_continual_steps_to_threshold_inf_when_bar_or_phase_missing():
    series = {
        "phase":           ["A",  "A"],
        "rec_step":        [0,    100],
        "test_accuracy_A": [0.50, 0.60],   # task A never reaches 0.80
        "test_accuracy_B": [0.50, 0.50],
    }
    m = metrics.continual_metrics(series)
    assert math.isinf(m["a_steps_to_80"])   # bar never cleared in phase A
    assert math.isinf(m["b_steps_to_80"])   # phase B never occurs


def test_continual_metrics_nan_when_phase_missing():
    # No A+B phase recorded -> consolidation/relearn are nan, not a crash.
    series = {
        "phase":           ["A",  "A",  "B",  "B"],
        "test_accuracy_A": [0.50, 0.95, 0.60, 0.40],
        "test_accuracy_B": [0.50, 0.50, 0.70, 0.95],
    }
    m = metrics.continual_metrics(series)
    assert m["forgetting"] == pytest.approx(0.55)
    assert math.isnan(m["consolidation"])


# -- compute cost: grow-scan candidate counts --------------------------------

def test_ghost_scan_cost_scored_le_dense_and_dense_matches_bruteforce():
    net = build_graph([2, 4, 4, 2], density=0.5, seed=7)
    init_weights(net, seed=7)
    X, y = generate_blobs(n=16, seed=4)
    out = metrics.ghost_scan_cost(net, X, y)
    brute_dense = sum(1 for j in range(net.num_neurons)
                      for i in range(net.num_neurons)
                      if net.neurons[i].layer < net.neurons[j].layer
                      and (i, j) not in net.synapses)
    assert out["ghost_dense_cost"] == float(brute_dense)
    assert 0.0 <= out["ghost_pairs_scored"] <= out["ghost_dense_cost"]


def test_ghost_scan_cost_empty_set_is_zero_scored():
    net = build_graph([2, 3, 2], density=0.5, seed=8)
    out = metrics.ghost_scan_cost(net, [], [])
    assert out["ghost_pairs_scored"] == 0.0
    assert out["ghost_dense_cost"] > 0.0


def test_cost_metrics_are_registered_and_in_compute_family():
    for k in ("ghost_dense_cost", "ghost_pairs_scored",
              "train_wall_time_sec", "wall_ms_per_step",
              "edge_steps_per_sec", "train_edge_steps",
              "edge_steps_to_90", "edge_steps_to_95",
              "avg_live_edges", "hidden_firing_frac",
              "fwd_active_edge_frac", "bwd_active_edge_frac",
              "grad_active_edge_frac"):
        assert k in metrics.METRIC_DESCRIPTIONS
    assert metrics.METRIC_DIRECTIONS["ghost_dense_cost"] == "neutral"
    assert metrics.METRIC_DIRECTIONS["edge_steps_to_95"] == "lower"
    assert metrics.METRIC_DIRECTIONS["edge_steps_per_sec"] == "higher"
    assert "train_edge_steps" in metrics.METRIC_FAMILIES["Compute cost"]
