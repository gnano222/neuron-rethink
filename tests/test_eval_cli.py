"""Tests for the CLI: argument parsing, spec building, and a tiny end-to-end."""

from __future__ import annotations

import os

import pytest

from evals import cli


def test_parse_args_defaults():
    args = cli.parse_args([])
    assert args.variants == "phasic-startle-k4,phasic-startle"
    assert args.seeds == 5
    assert args.steps == 15000          # promoted single-task horizon
    assert args.shift == 0
    assert args.baseline == "phasic-startle-k4"
    assert args.dataset == "spirals"


def test_cli_backend_flag():
    assert cli.build_spec(cli.parse_args([])).backend == "object"   # default
    args = cli.parse_args(["--variants", "phasic-startle-k4", "--baseline",
                           "phasic-startle-k4", "--backend", "array"])
    assert cli.build_spec(args).backend == "array"


def test_cli_accepts_digits_dataset():
    args = cli.parse_args(["--variants", "phasic-startle-k4,fully-connected",
                           "--dataset", "digits", "--layers", "64,64,32,10",
                           "--baseline", "phasic-startle-k4"])
    spec = cli.build_spec(args)
    assert spec.dataset == "digits"
    assert spec.layers == (64, 64, 32, 10)


def test_build_spec_parses_lists_and_layers():
    args = cli.parse_args(["--variants", "currency,core",
                           "--layers", "2,4,2", "--seeds", "3"])
    spec = cli.build_spec(args)
    assert spec.variants == ("currency", "core")
    assert spec.layers == (2, 4, 2)
    assert spec.seeds == 3


def test_build_spec_defaults_to_w16_topology():
    # with no --layers, the suite uses the promoted w16 default (the sweet spot
    # from the neuron-width sweep), not the old (2,10,10,8,2).
    spec = cli.build_spec(cli.parse_args([]))
    assert spec.layers == (2, 16, 16, 16, 2)


def test_main_rejects_unknown_variant():
    with pytest.raises(SystemExit):
        cli.main(["--variants", "currency,bogus", "--no-cache"])


def test_main_rejects_baseline_not_in_variants():
    with pytest.raises(SystemExit):
        cli.main(["--variants", "currency,core", "--baseline", "sleep",
                  "--no-cache"])


def test_main_end_to_end_tiny(tmp_path):
    agg = cli.main([
        "--variants", "currency,core", "--baseline", "core",
        "--seeds", "2", "--dataset", "blobs", "--steps", "300",
        "--record-every", "100", "--points", "80", "--layers", "2,4,4,2",
        "--jobs", "1", "--out", str(tmp_path), "--no-cache", "--n-boot", "200",
    ])
    assert set(agg["variants"]) == {"currency", "core"}
    assert os.path.exists(os.path.join(str(tmp_path), "scorecard.md"))
    assert os.path.exists(os.path.join(str(tmp_path), "acc_curves.png"))
    assert os.path.exists(os.path.join(str(tmp_path), "summary.txt"))


def test_parse_args_continual_defaults_to_single():
    args = cli.parse_args([])
    assert args.regime == "single"


def test_build_spec_passes_continual_fields():
    args = cli.parse_args(["--regime", "continual", "--steps-a", "120",
                           "--steps-b", "110", "--steps-ab", "70",
                           "--continual-turns", "0.7"])
    spec = cli.build_spec(args)
    assert spec.regime == "continual"
    assert spec.steps_a == 120 and spec.steps_b == 110 and spec.steps_ab == 70
    assert spec.continual_turns == 0.7


def test_main_end_to_end_continual_tiny(tmp_path):
    agg = cli.main([
        "--variants", "currency,core", "--baseline", "core",
        "--regime", "continual", "--steps-a", "120", "--steps-b", "120",
        "--steps-ab", "80", "--seeds", "2", "--record-every", "40",
        "--points", "60", "--layers", "2,4,4,2",
        "--jobs", "1", "--out", str(tmp_path), "--no-cache", "--n-boot", "200",
    ])
    assert "forgetting" in agg["metrics"]
    assert os.path.exists(os.path.join(str(tmp_path), "continual_curves.png"))


def test_main_publish_creates_run_folder(tmp_path):
    out = tmp_path / "out"
    pub = tmp_path / "pub"
    cli.main([
        "--variants", "currency,core", "--baseline", "core",
        "--seeds", "1", "--dataset", "blobs", "--steps", "300",
        "--record-every", "100", "--points", "80", "--layers", "2,4,4,2",
        "--jobs", "1", "--out", str(out), "--no-cache", "--n-boot", "100",
        "--publish", "--run-name", "demo", "--publish-dir", str(pub),
    ])
    run_dir = pub / "demo"
    assert (run_dir / "README.md").exists()
    assert (run_dir / "acc_curves.png").exists()
    assert (run_dir / "metrics.json").exists()
    readme = (run_dir / "README.md").read_text()
    assert "Key metrics" in readme and "demo" in readme
