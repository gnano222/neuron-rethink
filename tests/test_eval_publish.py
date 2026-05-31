"""Tests for the per-run publishing step (git-committable run folders)."""

from __future__ import annotations

import os

import pytest

from evals import publish
from evals.spec import SuiteSpec


def _synth_agg():
    return {
        "variants": ["legacy-full", "currency"],
        "baseline": "legacy-full",
        "directions": {"final_test_acc": "higher",
                       "max_grows_into_one_neuron": "lower",
                       "synapse_count_start": "neutral"},
        "metrics": {
            "final_test_acc": {
                "legacy-full": {"mean": 0.647, "std": 0.086},
                "currency": {"mean": 0.967, "std": 0.019, "verdict": "better"},
            },
            "max_grows_into_one_neuron": {
                "legacy-full": {"mean": 6.0, "std": 0.0},
                "currency": {"mean": 29.8, "std": 3.97, "verdict": "worse"},
            },
            "synapse_count_start": {                      # not a KEY_METRIC
                "legacy-full": {"mean": 102.0, "std": 1.4},
                "currency": {"mean": 104.0, "std": 1.4, "verdict": "≈"},
            },
        },
    }


def test_highlight_table_only_includes_key_metrics():
    table = publish.build_highlight_table(_synth_agg())
    assert "final_test_acc" in table
    assert "max_grows_into_one_neuron" in table
    assert "synapse_count_start" not in table     # filtered out (not key)
    assert "currency" in table and "legacy-full" in table
    assert "▲" in table and "▼" in table          # verdict markers carried


def test_continual_metrics_are_key_metrics():
    # forgetting / consolidation must be headline metrics so a continual run's
    # README leads with them.
    from evals.publish import KEY_METRICS
    assert "forgetting" in KEY_METRICS
    assert "consolidation" in KEY_METRICS


def test_highlight_table_includes_metric_descriptions():
    from evals.metrics import METRIC_DESCRIPTIONS
    table = publish.build_highlight_table(_synth_agg())
    assert "What it means" in table                       # description column
    assert METRIC_DESCRIPTIONS["final_test_acc"] in table  # the actual text
    # every key metric must have a description so the column is never blank
    from evals.publish import KEY_METRICS
    for k in KEY_METRICS:
        assert k in METRIC_DESCRIPTIONS and METRIC_DESCRIPTIONS[k]


def test_build_run_readme_has_metadata_table_and_images():
    md = publish.build_run_readme(
        _synth_agg(),
        meta={"run_name": "spirals_demo", "date": "2026-05-30 18:00:00",
              "seeds": 5, "dataset": "spirals", "steps": 30000, "shift": 6000,
              "git_sha": "abc1234", "command": "python evaluate.py ..."},
        image_names=["acc_curves.png", "verdict_heatmap.png"])
    assert "spirals_demo" in md
    assert "## Key metrics" in md
    assert "Prediction performance" in md          # full scorecard embedded
    assert "![acc_curves](acc_curves.png)" in md
    assert "![verdict_heatmap](verdict_heatmap.png)" in md
    assert "abc1234" in md


def test_publish_run_copies_artifacts_and_writes_readme(tmp_path):
    out = tmp_path / "run_out"
    out.mkdir()
    for name in ("acc_curves.png", "verdict_heatmap.png"):
        (out / name).write_bytes(b"\x89PNG\r\n")     # dummy png bytes
    (out / "scorecard.csv").write_text("family,metric\n")
    (out / "metrics.json").write_text("{}")
    (out / "summary.txt").write_text("table here")

    dest_root = tmp_path / "docs"
    dest = publish.publish_run(str(out), _synth_agg(), SuiteSpec(seeds=5),
                               dest_root=str(dest_root), run_id="spirals_demo",
                               command="python evaluate.py ...")

    assert os.path.basename(dest) == "spirals_demo"
    for name in ("README.md", "acc_curves.png", "verdict_heatmap.png",
                 "scorecard.csv", "metrics.json", "summary.txt"):
        assert os.path.exists(os.path.join(dest, name)), name
    readme = open(os.path.join(dest, "README.md")).read()
    assert "spirals_demo" in readme and "Key metrics" in readme
    assert "![acc_curves](acc_curves.png)" in readme
