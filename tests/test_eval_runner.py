"""Tests for the run driver: a well-formed RunResult, determinism, the shift
phase, the parallel pool, and the disk cache.

All runs here are tiny (few hundred steps, small net, few points) so the file
stays fast.
"""

from __future__ import annotations

import json
import os

import pytest

from evals.spec import SuiteSpec
from evals import runner, metrics


def tiny_spec(**kw):
    base = dict(variants=("currency", "core"), seeds=2, dataset="blobs",
                steps=300, record_every=100, layers=(2, 4, 4, 2), density=0.5,
                n_points=80, baseline="core")
    base.update(kw)
    return SuiteSpec(**base)


def test_run_one_returns_wellformed_result():
    spec = tiny_spec()
    res = runner.run_one("currency", seed=0, spec=spec)

    assert res["variant"] == "currency" and res["seed"] == 0
    rec = res["series"]["rec_step"]
    assert len(rec) >= 2
    for key in ("train_accuracy", "test_accuracy", "test_loss", "synapse_count",
                "mean_confidence", "cum_grow", "cum_prune"):
        assert len(res["series"][key]) == len(rec), key

    # every final metric is a known one; the core schema keys are present
    assert set(res["final"]) <= set(metrics.METRIC_DIRECTIONS)
    for key in ("final_test_acc", "max_test_acc", "auc_test_acc", "steps_to_90",
                "final_acc_stability", "n_grow_events", "turnover",
                "max_grows_into_one_neuron", "oscillation_frac", "max_regrow",
                "p10_utility", "freeloader_frac", "conf_utility_corr",
                "dead_unit_count", "dead_unit_frac", "mean_neuron_activation",
                "effective_density", "mean_survivor_age"):
        assert key in res["final"], key

    # distributions for the diagnostic plots
    for key in ("utilities", "confidences", "ages", "pruned_lifespans"):
        assert key in res["dist"], key


def test_runner_honors_per_variant_init_density():
    # the fully-connected arm pins init_density=1.0, so its initial graph is
    # dense (all-to-all) regardless of the suite --density. For (2,4,4,2) the
    # full edge count is 2*4 + 4*4 + 4*2 = 32.
    spec = tiny_spec(variants=("fully-connected", "currency"),
                     baseline="currency", layers=(2, 4, 4, 2), density=0.5,
                     steps=4, record_every=2)
    fc = runner.run_one("fully-connected", 0, spec)
    assert fc["final"]["synapse_count_start"] == 32
    assert fc["final"]["effective_density"] == 1.0
    # the sparse arm at the SAME suite density starts well below fully connected
    cur = runner.run_one("currency", 0, spec)
    assert cur["final"]["synapse_count_start"] < 32


def test_runner_honors_per_variant_init_layers():
    # a size-sweep arm pins its own neuron counts via init_layers, overriding the
    # suite --layers. The built graph must match build_graph on the OVERRIDE layers
    # (not the suite's), at the suite density/seed.
    from sprout.network import build_graph
    spec = tiny_spec(variants=("size-w4", "size-w10"), baseline="size-w10",
                     layers=(2, 8, 8, 2), density=0.5, steps=4, record_every=2)
    res = runner.run_one("size-w4", 0, spec)
    expected = build_graph([2, 4, 4, 4, 2], density=spec.density, seed=0)
    assert res["initial_edge_count"] == len(expected.synapses)
    # and it is NOT the suite-layer graph (proves the override actually fired)
    suite_graph = build_graph([2, 8, 8, 2], density=spec.density, seed=0)
    assert res["initial_edge_count"] != len(suite_graph.synapses)


def test_run_one_is_deterministic():
    spec = tiny_spec()
    a = runner.run_one("currency", 0, spec)
    b = runner.run_one("currency", 0, spec)
    assert a["final"]["final_test_acc"] == b["final"]["final_test_acc"]
    assert a["series"]["test_accuracy"] == b["series"]["test_accuracy"]


def test_no_shift_omits_recovery_metrics():
    res = runner.run_one("currency", 0, tiny_spec())
    assert res["shift_start_index"] is None
    assert "recovery_steps" not in res["final"]


def test_shift_adds_recovery_metrics():
    res = runner.run_one("currency", 0, tiny_spec(shift_steps=200))
    assert res["shift_start_index"] is not None
    for key in ("pre_shift_test_acc", "recovered_test_acc", "recovery_gap",
                "recovery_steps"):
        assert key in res["final"], key


def test_run_suite_serial_cartesian():
    results = runner.run_suite(tiny_spec(), jobs=1, cache_dir=None,
                               use_cache=False)
    assert len(results) == 4
    assert {(r["variant"], r["seed"]) for r in results} == {
        ("currency", 0), ("currency", 1), ("core", 0), ("core", 1)}


def test_cache_round_trip(tmp_path):
    spec = tiny_spec(variants=("core",), seeds=1)
    cache = str(tmp_path)
    first = runner.run_suite(spec, jobs=1, cache_dir=cache, use_cache=True)
    files = [f for f in os.listdir(cache) if f.endswith(".json")]
    assert len(files) == 1
    second = runner.run_suite(spec, jobs=1, cache_dir=cache, use_cache=True)
    # compare via JSON form so legitimate NaN metrics (NaN != NaN) still match
    assert (json.dumps(first[0]["final"], sort_keys=True)
            == json.dumps(second[0]["final"], sort_keys=True))


def test_parallel_matches_serial():
    spec = tiny_spec()
    serial = runner.run_suite(spec, jobs=1, cache_dir=None, use_cache=False)
    parallel = runner.run_suite(spec, jobs=2, cache_dir=None, use_cache=False)

    def by_key(rs):
        return {(r["variant"], r["seed"]): r["final"]["final_test_acc"]
                for r in rs}

    assert by_key(serial) == by_key(parallel)


# -- continual (forgetting) regime -------------------------------------------

def tiny_continual_spec(**kw):
    base = dict(variants=("currency", "core"), seeds=2, dataset="spirals",
                regime="continual", steps_a=120, steps_b=120, steps_ab=80,
                record_every=40, layers=(2, 4, 4, 2), density=0.5, n_points=60,
                baseline="core")
    base.update(kw)
    return SuiteSpec(**base)


def test_run_one_continual_wellformed():
    spec = tiny_continual_spec()
    res = runner.run_one_continual("currency", seed=0, spec=spec)

    assert res["variant"] == "currency" and res["regime"] == "continual"
    s = res["series"]
    rec = s["rec_step"]
    assert len(rec) >= 3
    for key in ("phase", "test_accuracy_A", "test_accuracy_B", "test_accuracy",
                "train_accuracy", "synapse_count", "mean_confidence",
                "cum_grow", "cum_prune"):
        assert len(s[key]) == len(rec), key

    # phases run in order A -> B -> AB
    phases = s["phase"]
    assert phases[0] == "A"
    assert phases.index("A") < phases.index("B") < phases.index("AB")

    # every final metric is known; continual + structural metrics both present
    assert set(res["final"]) <= set(metrics.METRIC_DIRECTIONS)
    for key in ("a_peak", "b_learned", "forgetting", "consolidation",
                "relearn_gap", "final_test_acc", "conf_utility_corr"):
        assert key in res["final"], key


def test_run_one_continual_is_deterministic():
    spec = tiny_continual_spec()
    a = runner.run_one_continual("currency", 0, spec)
    b = runner.run_one_continual("currency", 0, spec)
    assert a["series"]["test_accuracy_A"] == b["series"]["test_accuracy_A"]
    assert a["series"]["test_accuracy_B"] == b["series"]["test_accuracy_B"]


def test_run_suite_dispatches_to_continual():
    spec = tiny_continual_spec(variants=("core",), seeds=1)
    results = runner.run_suite(spec, jobs=1, cache_dir=None, use_cache=False)
    assert len(results) == 1
    assert results[0]["regime"] == "continual"
    assert "forgetting" in results[0]["final"]
