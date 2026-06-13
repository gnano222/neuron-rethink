"""Pure metric functions for the evaluation harness.

Every function here is side-effect free (it may run forward/backward passes on a
network, but it does not mutate persistent state beyond transient activations).
They fall into the four families from the design spec — prediction performance,
training efficacy, synapse structure, synapse quality — plus a couple of shared
primitives (fresh demand, event-log parsing).
"""

from __future__ import annotations

import math

import numpy as np

EPS = 1e-9


# ---------------------------------------------------------------------------
# A. prediction performance
# ---------------------------------------------------------------------------

def test_loss(net, X, y) -> float:
    """Mean cross-entropy of the softmax outputs over a labelled set."""
    if len(X) == 0:
        return 0.0
    total = 0.0
    for xi, yi in zip(X, y):
        probs = net.forward(xi)
        total += -math.log(float(probs[int(yi)]) + 1e-12)
    return total / len(X)


# ---------------------------------------------------------------------------
# shared: fresh gradient demand (fair across architectures)
# ---------------------------------------------------------------------------

def fresh_demand(net, X, y) -> dict:
    """Per-synapse mean ``|dL/dw|`` over a labelled set, measured fresh.

    This is the architecture-neutral stand-in for the metered ``grad_mag``: it
    works for legacy variants (which never populate ``grad_mag``) too, so utility
    and calibration are comparable across every variant.
    """
    acc = {k: 0.0 for k in net.synapses}
    n = 0
    for xi, yi in zip(X, y):
        net.forward(xi)
        _, gw, _ = net.backward(int(yi))
        for k in net.synapses:
            acc[k] += abs(gw[k])
        n += 1
    if n == 0:
        return acc
    return {k: v / n for k, v in acc.items()}


# ---------------------------------------------------------------------------
# D. synapse quality — utility / value
# ---------------------------------------------------------------------------

def synapse_utilities(net, demand: dict, lam: float = 1.0) -> dict:
    """``u = |w|/mean|w| + lam * demand/mean(demand)`` per surviving synapse."""
    edges = list(net.synapses)
    if not edges:
        return {}
    abs_w = np.array([abs(net.synapses[k].weight) for k in edges])
    wbar = abs_w.mean()
    wbar = wbar if wbar > EPS else 1.0
    dvals = np.array([demand.get(k, 0.0) for k in edges])
    dbar = dvals.mean()
    dbar = dbar if dbar > EPS else 1.0
    return {k: abs(net.synapses[k].weight) / wbar + lam * demand.get(k, 0.0) / dbar
            for k in edges}


def utility_stats(utilities: dict, prune_u_floor: float = 0.5) -> dict:
    if not utilities:
        return {"mean_utility": 0.0, "p10_utility": 0.0, "freeloader_frac": 0.0}
    vals = np.array(list(utilities.values()), dtype=float)
    return {
        "mean_utility": float(vals.mean()),
        "p10_utility": float(np.percentile(vals, 10)),
        "freeloader_frac": float(np.mean(vals < prune_u_floor)),
    }


# ---------------------------------------------------------------------------
# D. synapse quality — confidence calibration
# ---------------------------------------------------------------------------

def confidence_calibration(net, utilities: dict, conf_threshold: float = 1.0,
                           prune_u_floor: float = 0.5) -> dict:
    edges = [k for k in net.synapses if k in utilities]
    if len(edges) < 2:
        return {"conf_utility_corr": float("nan"), "frozen_freeloader_frac": 0.0}
    confs = np.array([net.synapses[k].confidence for k in edges])
    us = np.array([utilities[k] for k in edges])
    if confs.std() < EPS or us.std() < EPS:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(confs, us)[0, 1])
    frozen_freeloader = np.mean((confs > conf_threshold) & (us < prune_u_floor))
    return {"conf_utility_corr": corr,
            "frozen_freeloader_frac": float(frozen_freeloader)}


# ---------------------------------------------------------------------------
# D. synapse quality — effective capacity
# ---------------------------------------------------------------------------

def dead_unit_count(net, X, eps: float = EPS) -> int:
    """Number of hidden neurons whose ReLU never fires over ``X`` (max act ~0)."""
    last = len(net.layers) - 1
    hidden = [nid for L in range(1, last) for nid in net.layers[L]]
    if not hidden:
        return 0
    max_act = {nid: 0.0 for nid in hidden}
    for xi in X:
        net.forward(xi)
        for nid in hidden:
            a = net.neurons[nid].activation
            if a > max_act[nid]:
                max_act[nid] = a
    return sum(1 for nid in hidden if max_act[nid] < eps)


def neuron_activation_stats(net, X, eps: float = EPS) -> dict:
    """The "average neuron value": how much hidden neurons actually output.

    ``mean_neuron_activation`` is each hidden neuron's mean ReLU activation over
    ``X``, then averaged over neurons (equal weight per neuron) — so it stays
    comparable as the layer width changes. ``dead_unit_frac`` is the fraction of
    hidden neurons that never fire (the scale-free companion to the absolute
    ``dead_unit_count``). ``idle_unit_frac`` is the honest capacity readout:
    hidden neurons that never fire OR have no outgoing wires — recycling turns
    corpses into firing blanks (so they stop counting as *dead*), but a blank
    stays *idle* until the grow market actually rehires it. All are 0.0 for a
    net with no hidden layer.
    """
    last = len(net.layers) - 1
    hidden = [nid for L in range(1, last) for nid in net.layers[L]]
    if not hidden or len(X) == 0:
        return {"mean_neuron_activation": 0.0, "dead_unit_frac": 0.0,
                "idle_unit_frac": 0.0}
    sum_act = {nid: 0.0 for nid in hidden}
    max_act = {nid: 0.0 for nid in hidden}
    for xi in X:
        net.forward(xi)
        for nid in hidden:
            a = net.neurons[nid].activation
            sum_act[nid] += a
            if a > max_act[nid]:
                max_act[nid] = a
    per_neuron_mean = [sum_act[nid] / len(X) for nid in hidden]
    dead = sum(1 for nid in hidden if max_act[nid] < eps)
    idle = sum(1 for nid in hidden
               if max_act[nid] < eps or not net.outgoing[nid])
    return {
        "mean_neuron_activation": float(np.mean(per_neuron_mean)),
        "dead_unit_frac": dead / len(hidden),
        "idle_unit_frac": idle / len(hidden),
    }


# ---------------------------------------------------------------------------
# E. compute cost — grow-scan candidate counts
# ---------------------------------------------------------------------------

def ghost_scan_cost(net, X, y, grow_demand_k=None) -> dict:
    """Compute cost of one grow scan on this net: candidate ghost wires.

    ``ghost_dense_cost`` is the activity-independent count of valid candidates
    (~N^2); ``ghost_pairs_scored`` is the mean per-sample count actually evaluated
    after activity + demand pruning. Reuses the *same* candidate helpers the
    trainer's grow scan uses, so the measurement cannot drift from behaviour.
    """
    from sprout.currency import (active_ghost_sets, iter_ghost_candidates,
                                  dense_ghost_count)
    dense = float(dense_ghost_count(net))
    if len(X) == 0:
        return {"ghost_pairs_scored": 0.0, "ghost_dense_cost": dense}
    total = 0
    for xi, yi in zip(X, y):
        net.forward(xi)
        _, _, grad_b = net.backward(int(yi))
        ap, apo = active_ghost_sets(net, grad_b, grow_demand_k)
        total += sum(1 for _ in iter_ghost_candidates(net, ap, apo))
    return {"ghost_pairs_scored": total / len(X), "ghost_dense_cost": dense}


# ---------------------------------------------------------------------------
# C. synapse structure — event-log derived
# ---------------------------------------------------------------------------

def _edge_event_sequences(events) -> dict:
    seqs: dict = {}
    for e in events:
        seqs.setdefault(e["edge"], []).append((e["step"], e["type"]))
    for k in seqs:
        seqs[k].sort(key=lambda st: st[0])
    return seqs


def structural_metrics(events) -> dict:
    grow = [e for e in events if e["type"] == "grow"]
    prune = [e for e in events if e["type"] == "prune"]
    grows_per_target: dict = {}
    for e in grow:
        j = e["edge"][1]
        grows_per_target[j] = grows_per_target.get(j, 0) + 1
    return {
        "n_grow_events": len(grow),
        "n_prune_events": len(prune),
        "n_startle_events": sum(1 for e in events if e["type"] == "startle"),
        "distinct_neurons_grown": len(grows_per_target),
        "max_grows_into_one_neuron": max(grows_per_target.values(), default=0),
    }


# ---------------------------------------------------------------------------
# D. synapse quality — stability / lifespan (the open problem)
# ---------------------------------------------------------------------------

def oscillation_metrics(events) -> dict:
    seqs = _edge_event_sequences(events)
    grow_counts = {edge: sum(1 for _, t in seq if t == "grow")
                   for edge, seq in seqs.items()}
    grown = [e for e, c in grow_counts.items() if c >= 1]
    multi = [e for e, c in grow_counts.items() if c >= 2]
    return {
        "oscillation_frac": (len(multi) / len(grown)) if grown else 0.0,
        "max_regrow": max((c - 1 for c in grow_counts.values()), default=0),
        "n_distinct_grown": len(grown),
    }


def pruned_lifespans(events, initial_edges) -> list:
    """Lifespan (steps lived) of every synapse that was pruned.

    Birth is the most recent grow before the prune, or step 0 if the edge was
    present at build time (in ``initial_edges``).
    """
    initial_edges = set(initial_edges)
    seqs = _edge_event_sequences(events)
    lifespans = []
    for edge, seq in seqs.items():
        last_birth = 0 if edge in initial_edges else None
        for step, typ in seq:
            if typ == "grow":
                last_birth = step
            elif typ == "prune":
                birth = last_birth if last_birth is not None else 0
                lifespans.append(step - birth)
                last_birth = None
    return lifespans


# ---------------------------------------------------------------------------
# B. training efficacy — pure over the recorded series
# ---------------------------------------------------------------------------

def steps_to_threshold(rec_step, acc_series, threshold: float) -> float:
    for s, a in zip(rec_step, acc_series):
        if a >= threshold:
            return float(s)
    return math.inf


def auc(rec_step, acc_series) -> float:
    """Trapezoidal area under the accuracy curve, normalised to [0,1] in x."""
    xs = np.asarray(rec_step, dtype=float)
    ys = np.asarray(acc_series, dtype=float)
    if xs.size < 2:
        return float(ys[0]) if ys.size else 0.0
    span = xs[-1] - xs[0]
    if span <= 0:
        return float(ys.mean())
    area = float(np.sum((ys[1:] + ys[:-1]) / 2.0 * np.diff(xs)))
    return area / span


def stability(acc_series, k: int = 10) -> float:
    if not len(acc_series):
        return 0.0
    tail = acc_series[-k:]
    return float(np.std(tail))


def recovery_metrics(rec_step, acc_series, shift_start_index: int) -> dict:
    """Concept-shift recovery: how far accuracy fell and how long to climb back.

    ``shift_start_index`` is the index of the first post-shift record. The bar to
    regain is the last pre-shift accuracy; ``recovery_steps`` is measured from the
    first post-shift record step.
    """
    n = len(acc_series)
    if shift_start_index <= 0 or shift_start_index >= n:
        nan = float("nan")
        return {"pre_shift_acc": nan, "recovered_acc": nan,
                "recovery_gap": nan, "recovery_steps": math.inf}
    pre = float(acc_series[shift_start_index - 1])
    t0 = rec_step[shift_start_index]
    recovered = float(acc_series[-1])
    rec_steps = math.inf
    for s, a in zip(rec_step[shift_start_index:], acc_series[shift_start_index:]):
        if a >= pre:
            rec_steps = float(s - t0)
            break
    return {"pre_shift_acc": pre, "recovered_acc": recovered,
            "recovery_gap": pre - recovered, "recovery_steps": rec_steps}


def phase_steps_to_threshold(series, phase_label, track_key,
                             threshold: float) -> float:
    """Steps *within a phase* to first reach ``threshold`` on a task's held-out
    accuracy track, measured from that phase's first recorded step.

    This is the per-task learning *speed*: how quickly a model reaches a given
    accuracy bar once a task's phase begins. ``inf`` if the bar is never cleared
    in the phase (or the phase never occurs).
    """
    rec = series.get("rec_step", [])
    phase = series.get("phase", [])
    track = series.get(track_key, [])
    n = min(len(rec), len(phase), len(track))
    t0 = None
    for i in range(n):
        if phase[i] != phase_label:
            continue
        if t0 is None:
            t0 = rec[i]                 # this phase's first recorded step
        if track[i] >= threshold:
            return float(rec[i] - t0)
    return math.inf


def continual_metrics(series) -> dict:
    """Forgetting / consolidation from a continual run's dual-task series.

    Reads the per-snapshot ``phase`` label ("A"/"B"/"AB") and the two held-out
    accuracy tracks. Measured at phase boundaries so each signal stays clean:
    ``forgetting`` is read at the A->B boundary, *before* any consolidation.

        a_peak        = A accuracy at the end of phase A
        b_learned     = B accuracy at the end of phase B (forward learning)
        forgetting    = a_peak - A accuracy at the end of phase B   (lower=better)
        consolidation = min(A, B) at the end of phase A+B           (can it hold both)
        relearn_gap   = a_peak - A accuracy at the end of phase A+B
        {a,b}_steps_to_{80,90} = steps into each task's phase to reach that bar
                                 (per-task learning speed; lower=faster)
    """
    nan = float("nan")
    phase = series.get("phase", [])
    a = series.get("test_accuracy_A", [])
    b = series.get("test_accuracy_B", [])

    def last_in(ph, track):
        vals = [track[i] for i in range(min(len(phase), len(track)))
                if phase[i] == ph]
        return float(vals[-1]) if vals else nan

    def diff(x, y):
        return (x - y) if not (math.isnan(x) or math.isnan(y)) else nan

    a_peak = last_in("A", a)
    a_after_b = last_in("B", a)
    b_learned = last_in("B", b)
    cons_a = last_in("AB", a)
    cons_b = last_in("AB", b)
    consolidation = (min(cons_a, cons_b)
                     if not (math.isnan(cons_a) or math.isnan(cons_b)) else nan)

    out = {
        "a_peak": a_peak,
        "b_learned": b_learned,
        "forgetting": diff(a_peak, a_after_b),
        "consolidation": consolidation,
        "relearn_gap": diff(a_peak, cons_a),
    }
    # per-task learning speed: steps into each phase to reach the bar
    for thr in (0.80, 0.90):
        tag = f"{int(round(thr * 100))}"
        out[f"a_steps_to_{tag}"] = phase_steps_to_threshold(
            series, "A", "test_accuracy_A", thr)
        out[f"b_steps_to_{tag}"] = phase_steps_to_threshold(
            series, "B", "test_accuracy_B", thr)
    return out


# ---------------------------------------------------------------------------
# C. synapse structure — fan / density (end state)
# ---------------------------------------------------------------------------

def fan_stats(net) -> dict:
    last = len(net.layers) - 1
    fan_in = [len(net.incoming[nid]) for nid in range(net.num_neurons)
              if net.neurons[nid].layer != 0]
    fan_out = [len(net.outgoing[nid]) for nid in range(net.num_neurons)
               if net.neurons[nid].layer != last]
    full = sum(len(net.layers[L]) * len(net.layers[L + 1]) for L in range(last))
    return {
        "mean_fan_in": float(np.mean(fan_in)) if fan_in else 0.0,
        "mean_fan_out": float(np.mean(fan_out)) if fan_out else 0.0,
        "effective_density": (len(net.synapses) / full) if full else 0.0,
    }


# ---------------------------------------------------------------------------
# D. synapse quality — stability / lifespan (survivor side)
# ---------------------------------------------------------------------------

def survivor_age_stats(net) -> dict:
    ages = [s.age for s in net.synapses.values()]
    if not ages:
        return {"mean_survivor_age": 0.0, "median_survivor_age": 0.0}
    return {"mean_survivor_age": float(np.mean(ages)),
            "median_survivor_age": float(np.median(ages))}


# ---------------------------------------------------------------------------
# D. synapse quality — effective capacity (assembled)
# ---------------------------------------------------------------------------

def capacity_metrics(net, X, initial_count: int, eps: float = EPS) -> dict:
    weights = [abs(s.weight) for s in net.synapses.values()]
    inert = float(np.mean([w < eps for w in weights])) if weights else 0.0
    used = (len(net.synapses) / initial_count) if initial_count else 0.0
    return {
        "dead_unit_count": dead_unit_count(net, X, eps),
        "inert_synapse_frac": inert,
        "used_vs_allocated": used,
    }


# ---------------------------------------------------------------------------
# D. synapse quality — recycling (sleep-time corpse re-entry)
# ---------------------------------------------------------------------------

def recycle_metrics(events, net, X, eps: float = EPS) -> dict:
    """Recycling activity + outcome.

    ``n_recycle_events`` counts recycle firings (a unit can be recycled more
    than once). ``recycled_rehired_frac`` is the end-state outcome: of the
    *distinct* units ever recycled, the fraction ending the run non-idle —
    firing AND with outgoing wires, i.e. actually back in service. NaN when
    nothing was ever recycled (variants without the mechanism).
    """
    recycled = {e["neuron"] for e in events if e["type"] == "recycle"}
    if not recycled:
        return {"n_recycle_events": 0, "recycled_rehired_frac": float("nan")}
    max_act = {nid: 0.0 for nid in recycled}
    for xi in X:
        net.forward(xi)
        for nid in recycled:
            a = net.neurons[nid].activation
            if a > max_act[nid]:
                max_act[nid] = a
    rehired = sum(1 for nid in recycled
                  if max_act[nid] >= eps and net.outgoing[nid])
    return {
        "n_recycle_events": sum(1 for e in events if e["type"] == "recycle"),
        "recycled_rehired_frac": rehired / len(recycled),
    }


# ---------------------------------------------------------------------------
# sanity: metered signal fidelity (currency only)
# ---------------------------------------------------------------------------

def meter_fidelity(net, demand: dict) -> float:
    """Correlation of the metered ``grad_mag`` with the fresh gradient demand."""
    edges = list(net.synapses)
    if len(edges) < 2:
        return float("nan")
    m = np.array([net.synapses[k].grad_mag for k in edges])
    d = np.array([demand.get(k, 0.0) for k in edges])
    if m.std() < EPS or d.std() < EPS:
        return float("nan")
    return float(np.corrcoef(m, d)[0, 1])


# ---------------------------------------------------------------------------
# scorecard assembly
# ---------------------------------------------------------------------------

# Each scorecard metric is tagged so the bootstrap verdict reads correctly:
# "higher" = higher is better, "lower" = lower is better, "neutral" = descriptive.
METRIC_DIRECTIONS: dict[str, str] = {
    # A. prediction performance
    "final_test_acc": "higher",
    "max_test_acc": "higher",
    "final_train_acc": "higher",
    "final_test_loss": "lower",
    # B. training efficacy
    "steps_to_90": "lower",
    "steps_to_95": "lower",
    "auc_test_acc": "higher",
    "final_acc_stability": "lower",
    "pre_shift_test_acc": "higher",
    "recovered_test_acc": "higher",
    "recovery_gap": "lower",
    "recovery_steps": "lower",
    # B. continual learning (forgetting regime)
    "a_peak": "higher",
    "b_learned": "higher",
    "forgetting": "lower",
    "consolidation": "higher",
    "relearn_gap": "lower",
    "a_steps_to_80": "lower",
    "a_steps_to_90": "lower",
    "b_steps_to_80": "lower",
    "b_steps_to_90": "lower",
    # C. synapse structure
    "synapse_count_start": "neutral",
    "synapse_count_peak": "neutral",
    "synapse_count_end": "neutral",
    "n_grow_events": "neutral",
    "n_prune_events": "neutral",
    "n_startle_events": "neutral",
    "distinct_neurons_grown": "neutral",
    "turnover": "lower",
    "max_grows_into_one_neuron": "lower",
    "mean_fan_in": "neutral",
    "mean_fan_out": "neutral",
    "effective_density": "neutral",
    # D. synapse quality
    # (mean utility is intentionally omitted: |w|/mean|w| + lam*d/mean(d) has a
    # mean of exactly 1+lam by construction, so only its distribution informs)
    "p10_utility": "higher",
    "freeloader_frac": "lower",
    "mean_survivor_age": "higher",
    "median_survivor_age": "higher",
    "mean_pruned_lifespan": "neutral",
    "oscillation_frac": "lower",
    "max_regrow": "lower",
    "conf_utility_corr": "higher",
    "frozen_freeloader_frac": "lower",
    "dead_unit_count": "lower",
    "dead_unit_frac": "lower",
    "idle_unit_frac": "lower",
    "mean_neuron_activation": "neutral",
    "inert_synapse_frac": "lower",
    "used_vs_allocated": "neutral",
    "n_recycle_events": "neutral",
    "recycled_rehired_frac": "neutral",
    # sanity (currency)
    "meter_fidelity": "higher",
    # E. compute cost (descriptive: the scaling story is the chart, not verdicts)
    "ghost_dense_cost": "neutral",
    "ghost_pairs_scored": "neutral",
}

# one-line plain-language description per metric (shown in the key-metrics
# table while the reader is still learning the schema)
METRIC_DESCRIPTIONS: dict[str, str] = {
    # A. prediction performance
    "final_test_acc": "held-out accuracy at the end of the run",
    "max_test_acc": "best held-out accuracy reached at any point",
    "final_train_acc": "training-set accuracy at the end",
    "final_test_loss": "held-out cross-entropy at the end (lower=better)",
    # B. training efficacy
    "steps_to_90": "steps to first reach 90% test accuracy",
    "steps_to_95": "steps to first reach 95% test accuracy",
    "auc_test_acc": "area under the test-accuracy curve (speed + level)",
    "final_acc_stability": "jitter (std) of test accuracy over the last records",
    "pre_shift_test_acc": "test accuracy just before the concept shift",
    "recovered_test_acc": "test accuracy at the end, after the label swap",
    "recovery_gap": "accuracy lost vs pre-shift and not yet regained",
    "recovery_steps": "steps after the shift to return to pre-shift accuracy",
    # B. continual learning (forgetting regime)
    "a_peak": "accuracy on task A at the end of phase A (its peak)",
    "b_learned": "accuracy on task B at the end of phase B (forward learning)",
    "forgetting": "task A accuracy lost while learning B (lower=better)",
    "consolidation": "min(A, B) accuracy after interleaved A+B (holds both?)",
    "relearn_gap": "task A accuracy not restored by the A+B phase",
    "a_steps_to_80": "steps into phase A to reach 80% on task A (first-task speed)",
    "a_steps_to_90": "steps into phase A to reach 90% on task A (first-task speed)",
    "b_steps_to_80": "steps into phase B to reach 80% on task B (second-task speed)",
    "b_steps_to_90": "steps into phase B to reach 90% on task B (second-task speed)",
    # C. synapse structure
    "synapse_count_start": "live synapses at initialization",
    "synapse_count_peak": "max live synapses during the run",
    "synapse_count_end": "live synapses at the end",
    "n_grow_events": "total synapses grown over the run",
    "n_prune_events": "total synapses pruned over the run",
    "n_startle_events": "demand-spike hiring alarms fired (startle growth)",
    "distinct_neurons_grown": "how many neurons received grown wires",
    "turnover": "(grows + prunes) per average synapse — rewiring rate",
    "max_grows_into_one_neuron": "most times one neuron was grown into (churn)",
    "mean_fan_in": "avg incoming connections per non-input neuron",
    "mean_fan_out": "avg outgoing connections per non-output neuron",
    "effective_density": "live edges as a fraction of fully-connected",
    # D. synapse quality
    "p10_utility": "10th-percentile synapse utility (weak-wire floor)",
    "freeloader_frac": "fraction of synapses below the prune-utility floor",
    "mean_survivor_age": "avg steps the surviving synapses have lived",
    "median_survivor_age": "median age of surviving synapses",
    "mean_pruned_lifespan": "avg steps a pruned synapse lived before being cut",
    "oscillation_frac": "fraction of grown edges grown ≥2× (thrash)",
    "max_regrow": "most times one edge was regrown after a prune",
    "conf_utility_corr": "corr of confidence with real utility (calibration)",
    "frozen_freeloader_frac": "high-confidence but low-utility synapses",
    "dead_unit_count": "hidden neurons that never fire on test data",
    "dead_unit_frac": "fraction of hidden neurons that never fire (scale-free)",
    "idle_unit_frac": "fraction of hidden neurons dead OR outputless (not in service)",
    "mean_neuron_activation": "avg hidden-neuron ReLU output on test data (neuron value)",
    "inert_synapse_frac": "fraction of synapses with ~zero weight",
    "used_vs_allocated": "live edges vs edges present at init",
    "n_recycle_events": "dead-unit recycles fired over the run (sleep recycling)",
    "recycled_rehired_frac": "of recycled units, fraction back in service at the end",
    # sanity
    "meter_fidelity": "corr of metered vs fresh gradient (currency only)",
    # E. compute cost
    "ghost_dense_cost": "candidate ghost wires the grow-scan must consider (~N²)",
    "ghost_pairs_scored": "candidate wires actually scored after activity+demand pruning",
}

# which metric belongs to which family, for grouped scorecard rendering
METRIC_FAMILIES: dict[str, tuple[str, ...]] = {
    "Prediction performance": (
        "final_test_acc", "max_test_acc", "final_train_acc", "final_test_loss"),
    "Training efficacy": (
        "steps_to_90", "steps_to_95", "auc_test_acc", "final_acc_stability",
        "pre_shift_test_acc", "recovered_test_acc", "recovery_gap",
        "recovery_steps"),
    "Continual learning": (
        "a_peak", "b_learned", "forgetting", "consolidation", "relearn_gap",
        "a_steps_to_80", "a_steps_to_90", "b_steps_to_80", "b_steps_to_90"),
    "Synapse structure": (
        "synapse_count_start", "synapse_count_peak", "synapse_count_end",
        "n_grow_events", "n_prune_events", "n_startle_events",
        "distinct_neurons_grown", "turnover", "max_grows_into_one_neuron",
        "mean_fan_in", "mean_fan_out", "effective_density"),
    "Synapse quality": (
        "p10_utility", "freeloader_frac", "mean_survivor_age",
        "median_survivor_age", "mean_pruned_lifespan", "oscillation_frac",
        "max_regrow", "conf_utility_corr", "frozen_freeloader_frac",
        "dead_unit_count", "dead_unit_frac", "idle_unit_frac",
        "mean_neuron_activation", "inert_synapse_frac", "used_vs_allocated",
        "n_recycle_events", "recycled_rehired_frac"),
    "Compute cost": ("ghost_dense_cost", "ghost_pairs_scored"),
    "Signal sanity": ("meter_fidelity",),
}


def final_snapshot(net, X_test, y_test, events, initial_edges, series,
                   shift_start_index, cfg):
    """Assemble the flat ``final`` scalar dict and a ``dist`` dict of per-synapse
    arrays (for the diagnostic plots). Utility/calibration use the fresh,
    architecture-neutral gradient demand so variants are comparable.
    """
    demand = fresh_demand(net, X_test, y_test)
    utils = synapse_utilities(net, demand, lam=cfg.lam_prune)
    ustats = utility_stats(utils, prune_u_floor=cfg.prune_u_floor)
    cal = confidence_calibration(net, utils, conf_threshold=1.0,
                                 prune_u_floor=cfg.prune_u_floor)
    struct = structural_metrics(events)
    osc = oscillation_metrics(events)
    lifes = pruned_lifespans(events, initial_edges)
    fans = fan_stats(net)
    cap = capacity_metrics(net, X_test, initial_count=len(initial_edges))
    nact = neuron_activation_stats(net, X_test)
    recyc = recycle_metrics(events, net, X_test)
    scan = ghost_scan_cost(net, X_test, y_test,
                           getattr(cfg, "grow_demand_k", None))
    ages = survivor_age_stats(net)

    rec = series["rec_step"]
    test_acc = series["test_accuracy"]
    sc = series["synapse_count"]
    mean_count = float(np.mean(sc)) if sc else 1.0
    denom = mean_count if mean_count > 0 else 1.0
    turnover = (struct["n_grow_events"] + struct["n_prune_events"]) / denom

    final = {
        "final_test_acc": float(test_acc[-1]) if test_acc else 0.0,
        "max_test_acc": float(max(test_acc)) if test_acc else 0.0,
        "final_train_acc": (float(series["train_accuracy"][-1])
                            if series["train_accuracy"] else 0.0),
        "final_test_loss": (float(series["test_loss"][-1])
                            if series["test_loss"] else 0.0),
        "steps_to_90": steps_to_threshold(rec, test_acc, 0.90),
        "steps_to_95": steps_to_threshold(rec, test_acc, 0.95),
        "auc_test_acc": auc(rec, test_acc),
        "final_acc_stability": stability(test_acc),
        "synapse_count_start": float(sc[0]) if sc else 0.0,
        "synapse_count_peak": float(max(sc)) if sc else 0.0,
        "synapse_count_end": float(sc[-1]) if sc else 0.0,
        "n_grow_events": struct["n_grow_events"],
        "n_prune_events": struct["n_prune_events"],
        "n_startle_events": struct["n_startle_events"],
        "distinct_neurons_grown": struct["distinct_neurons_grown"],
        "turnover": turnover,
        "max_grows_into_one_neuron": struct["max_grows_into_one_neuron"],
        "mean_fan_in": fans["mean_fan_in"],
        "mean_fan_out": fans["mean_fan_out"],
        "effective_density": fans["effective_density"],
        "p10_utility": ustats["p10_utility"],
        "freeloader_frac": ustats["freeloader_frac"],
        "mean_survivor_age": ages["mean_survivor_age"],
        "median_survivor_age": ages["median_survivor_age"],
        "mean_pruned_lifespan": float(np.mean(lifes)) if lifes else 0.0,
        "oscillation_frac": osc["oscillation_frac"],
        "max_regrow": osc["max_regrow"],
        "conf_utility_corr": cal["conf_utility_corr"],
        "frozen_freeloader_frac": cal["frozen_freeloader_frac"],
        "dead_unit_count": cap["dead_unit_count"],
        "dead_unit_frac": nact["dead_unit_frac"],
        "idle_unit_frac": nact["idle_unit_frac"],
        "mean_neuron_activation": nact["mean_neuron_activation"],
        "n_recycle_events": recyc["n_recycle_events"],
        "recycled_rehired_frac": recyc["recycled_rehired_frac"],
        "ghost_dense_cost": scan["ghost_dense_cost"],
        "ghost_pairs_scored": scan["ghost_pairs_scored"],
        "inert_synapse_frac": cap["inert_synapse_frac"],
        "used_vs_allocated": cap["used_vs_allocated"],
        "meter_fidelity": meter_fidelity(net, demand),
    }
    if shift_start_index is not None:
        rcov = recovery_metrics(rec, test_acc, shift_start_index)
        final["pre_shift_test_acc"] = rcov["pre_shift_acc"]
        final["recovered_test_acc"] = rcov["recovered_acc"]
        final["recovery_gap"] = rcov["recovery_gap"]
        final["recovery_steps"] = rcov["recovery_steps"]

    dist = {
        "utilities": [float(utils[k]) for k in net.synapses if k in utils],
        "confidences": [float(net.synapses[k].confidence) for k in net.synapses],
        "ages": [int(net.synapses[k].age) for k in net.synapses],
        "pruned_lifespans": [float(x) for x in lifes],
    }
    return final, dist
