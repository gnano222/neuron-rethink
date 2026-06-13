"""Tests for the scorecard builders (pure) and a smoke test of the full report
(files + plots) on a tiny real suite.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from evals import report, metrics, aggregate, runner
from evals.spec import SuiteSpec


def _synth_agg():
    return {
        "variants": ["legacy-full", "currency"],
        "baseline": "legacy-full",
        "directions": {"final_test_acc": "higher",
                       "max_grows_into_one_neuron": "lower"},
        "metrics": {
            "final_test_acc": {
                "legacy-full": {"mean": 0.998, "std": 0.001,
                                "ci_low": 0.997, "ci_high": 0.999, "n": 5},
                "currency": {"mean": 0.992, "std": 0.002,
                             "ci_low": 0.990, "ci_high": 0.994, "n": 5,
                             "diff_mean": -0.006, "diff_ci_low": -0.008,
                             "diff_ci_high": -0.004, "verdict": "worse"},
            },
            "max_grows_into_one_neuron": {
                "legacy-full": {"mean": 6.0, "std": 0.0,
                                "ci_low": 6.0, "ci_high": 6.0, "n": 5},
                "currency": {"mean": 18.0, "std": 1.0,
                             "ci_low": 17.0, "ci_high": 19.0, "n": 5,
                             "diff_mean": 12.0, "diff_ci_low": 11.0,
                             "diff_ci_high": 13.0, "verdict": "worse"},
            },
        },
    }


def test_fmt_handles_special_values():
    assert report._fmt(math.inf) == "∞"
    assert report._fmt(math.nan) == "—"
    assert report._fmt(None) == "—"
    assert report._fmt(6.0) == "6"
    assert report._fmt(0.99812).startswith("0.998")


def test_verdict_marker_mapping():
    assert report.verdict_marker("better") == "▲"
    assert report.verdict_marker("worse") == "▼"
    assert report.verdict_marker("≈") == "≈"
    assert report.verdict_marker("n/a") == "?"
    assert report.verdict_marker("") == ""


def test_build_markdown_has_variants_families_and_verdict():
    md = report.build_markdown(_synth_agg())
    assert "legacy-full" in md and "currency" in md
    assert "(baseline)" in md                      # baseline column marked
    assert "Prediction performance" in md          # family header rendered
    assert "Synapse structure" in md
    assert "final_test_acc" in md
    assert "▼" in md                               # the 'worse' verdict marker


def test_build_csv_one_row_per_metric_variant():
    csv = report.build_csv(_synth_agg())
    lines = [l for l in csv.splitlines() if l.strip()]
    header = lines[0].split(",")
    assert header[:4] == ["family", "metric", "direction", "variant"]
    # 2 metrics x 2 variants = 4 data rows
    assert len(lines) - 1 == 4
    assert any("currency" in l and "worse" in l for l in lines[1:])


def test_build_table_is_plain_text_with_values():
    txt = report.build_table(_synth_agg())
    assert "currency" in txt and "final_test_acc" in txt


def test_write_report_creates_files(tmp_path):
    spec = SuiteSpec(variants=("currency", "core"), seeds=2, dataset="blobs",
                     steps=300, record_every=100, layers=(2, 4, 4, 2),
                     density=0.5, n_points=80, baseline="core")
    results = runner.run_suite(spec, jobs=1, cache_dir=None, use_cache=False)
    agg = aggregate.aggregate_suite(results, baseline="core",
                                    directions=metrics.METRIC_DIRECTIONS,
                                    n_boot=200, rng=np.random.default_rng(0))
    paths = report.write_report(agg, results, str(tmp_path))

    for key in ("markdown", "csv", "json"):
        assert os.path.exists(paths[key]) and os.path.getsize(paths[key]) > 0

    for png in ("acc_curves.png", "count_curves.png", "churn_curves.png",
                "verdict_heatmap.png"):
        p = os.path.join(str(tmp_path), png)
        assert os.path.exists(p) and os.path.getsize(p) > 0
    # one quality figure per variant
    assert os.path.exists(os.path.join(str(tmp_path), "quality_currency.png"))
    assert os.path.exists(os.path.join(str(tmp_path), "quality_core.png"))


def test_write_report_continual_emits_dual_task_curve(tmp_path):
    spec = SuiteSpec(variants=("currency", "core"), seeds=2, dataset="spirals",
                     regime="continual", steps_a=120, steps_b=120, steps_ab=80,
                     record_every=40, layers=(2, 4, 4, 2), density=0.5,
                     n_points=60, baseline="core")
    results = runner.run_suite(spec, jobs=1, cache_dir=None, use_cache=False)
    agg = aggregate.aggregate_suite(results, baseline="core",
                                    directions=metrics.METRIC_DIRECTIONS,
                                    n_boot=200, rng=np.random.default_rng(0))
    report.write_report(agg, results, str(tmp_path))
    # the money chart: per-task A vs B accuracy with phase boundaries
    p = os.path.join(str(tmp_path), "continual_curves.png")
    assert os.path.exists(p) and os.path.getsize(p) > 0


# -- scaling chart (grow-scan cost vs network size) --------------------------

def _agg_with_cost():
    def cell(m, s):
        return {"mean": m, "std": s, "ci_low": m, "ci_high": m, "verdict": ""}
    return {
        "variants": ["size-w4", "size-w16"],
        "baseline": "size-w16",
        "directions": {"ghost_dense_cost": "neutral",
                       "ghost_pairs_scored": "neutral"},
        "metrics": {
            "ghost_dense_cost": {"size-w4": cell(60, 0), "size-w16": cell(900, 0)},
            "ghost_pairs_scored": {"size-w4": cell(12, 0), "size-w16": cell(40, 0)},
        },
    }


def test_plot_scaling_writes_png_when_cost_present(tmp_path):
    results = [{"variant": "size-w4", "n_neurons": 14, "series": {}, "dist": {}},
               {"variant": "size-w16", "n_neurons": 50, "series": {}, "dist": {}}]
    path = tmp_path / "cost_scaling.png"
    ok = report._plot_scaling(_agg_with_cost(), results, str(path))
    assert ok is True and path.exists()


def test_plot_scaling_skips_when_cost_absent(tmp_path):
    agg = {"variants": ["a"], "baseline": "a", "directions": {},
           "metrics": {"final_test_acc": {"a": {"mean": 1.0, "std": 0.0}}}}
    path = tmp_path / "cost_scaling.png"
    ok = report._plot_scaling(agg, [{"variant": "a", "n_neurons": 5}], str(path))
    assert ok is False and not path.exists()
