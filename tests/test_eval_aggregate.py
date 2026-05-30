"""Tests for cross-seed aggregation and the bootstrap significance verdict."""

from __future__ import annotations

import math

import numpy as np
import pytest

from evals import aggregate


def test_summarize_mean_std_and_ci_bracket_the_mean():
    s = aggregate.summarize([1.0, 2.0, 3.0, 4.0, 5.0])
    assert s["mean"] == pytest.approx(3.0)
    assert s["std"] == pytest.approx(np.std([1, 2, 3, 4, 5]))
    assert s["ci_low"] <= s["mean"] <= s["ci_high"]


def test_summarize_handles_inf():
    s = aggregate.summarize([1.0, math.inf, 2.0])
    assert s["mean"] == math.inf
    assert math.isnan(s["ci_low"]) and math.isnan(s["ci_high"])


def test_verdict_directions():
    # CI entirely above 0 => variant > baseline
    assert aggregate.verdict(0.2, 0.5, "higher") == "better"
    assert aggregate.verdict(0.2, 0.5, "lower") == "worse"
    # CI entirely below 0 => variant < baseline
    assert aggregate.verdict(-0.5, -0.2, "higher") == "worse"
    assert aggregate.verdict(-0.5, -0.2, "lower") == "better"
    # straddles 0 => no clear difference
    assert aggregate.verdict(-0.1, 0.3, "higher") == "≈"
    assert aggregate.verdict(-0.1, 0.3, "lower") == "≈"
    # neutral metric never claims a winner
    assert aggregate.verdict(0.2, 0.5, "neutral") == "≈"


def test_bootstrap_diff_detects_clear_higher_better_win():
    rng = np.random.default_rng(0)
    out = aggregate.bootstrap_diff([1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0],
                                   direction="higher", n_boot=1000, rng=rng)
    assert out["diff_mean"] == pytest.approx(1.0)
    assert out["ci_low"] > 0
    assert out["verdict"] == "better"


def test_bootstrap_diff_lower_is_better():
    rng = np.random.default_rng(0)
    # variant churns less than baseline; lower is better => "better"
    out = aggregate.bootstrap_diff([2.0, 2.0, 2.0], [9.0, 9.0, 9.0],
                                   direction="lower", n_boot=1000, rng=rng)
    assert out["diff_mean"] == pytest.approx(-7.0)
    assert out["verdict"] == "better"


def test_bootstrap_diff_no_clear_difference():
    rng = np.random.default_rng(0)
    out = aggregate.bootstrap_diff([0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0],
                                   direction="higher", n_boot=1000, rng=rng)
    assert out["verdict"] == "≈"


def test_bootstrap_diff_inf_is_not_applicable():
    rng = np.random.default_rng(0)
    out = aggregate.bootstrap_diff([math.inf, math.inf], [10.0, 12.0],
                                   direction="lower", n_boot=200, rng=rng)
    assert out["verdict"] == "n/a"


def test_aggregate_suite_groups_and_compares():
    results = [
        {"variant": "currency", "seed": 0, "final": {"acc": 0.99, "churn": 19}},
        {"variant": "currency", "seed": 1, "final": {"acc": 0.99, "churn": 17}},
        {"variant": "legacy-full", "seed": 0, "final": {"acc": 0.99, "churn": 6}},
        {"variant": "legacy-full", "seed": 1, "final": {"acc": 0.99, "churn": 6}},
    ]
    directions = {"acc": "higher", "churn": "lower"}
    agg = aggregate.aggregate_suite(results, baseline="legacy-full",
                                    directions=directions, n_boot=500,
                                    rng=np.random.default_rng(0))
    assert set(agg["variants"]) == {"currency", "legacy-full"}
    assert agg["baseline"] == "legacy-full"
    # baseline column has summary but no verdict
    assert "verdict" not in agg["metrics"]["churn"]["legacy-full"]
    # currency churns more than the baseline (lower is better) => "worse"
    assert agg["metrics"]["churn"]["currency"]["verdict"] == "worse"
    assert agg["metrics"]["churn"]["currency"]["mean"] == pytest.approx(18.0)
