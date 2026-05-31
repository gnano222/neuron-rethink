"""Tests for the variant registry and suite specification."""

from __future__ import annotations

import pytest

from evals.spec import VARIANTS, make_config, SuiteSpec
from sprout.train import Config


def test_legacy_full_variant_uses_eligibility_not_currency():
    cfg = make_config("legacy-full")
    assert isinstance(cfg, Config)
    assert cfg.enable_eligibility is True
    assert cfg.grad_currency is False
    assert cfg.enable_prune and cfg.enable_grow
    # the tuned spirals settings the baseline needs
    assert cfg.theta_prune == 0.001
    assert cfg.prune_warmup == 6000


def test_currency_variant_uses_gradient_currency():
    cfg = make_config("currency")
    assert cfg.grad_currency is True
    assert cfg.enable_eligibility is False
    assert cfg.enable_confidence and cfg.enable_prune and cfg.enable_grow
    # the promoted baseline: default currency now uses the calibrated 2D
    # confidence with the softened (sigmoid) settled cliff
    assert cfg.confidence_mode == "twod"
    assert cfg.settled_mode == "sigmoid"


def test_currency_tugofwar_variant_preserves_old_rule():
    # the prior tug-of-war confidence rule is kept as an explicit variant for
    # comparison even though it is no longer the default
    cfg = make_config("currency-tugofwar")
    assert cfg.grad_currency is True
    assert cfg.confidence_mode == "tugofwar"


def test_currency_2dconf_variant_uses_twod_confidence():
    cfg = make_config("currency-2dconf")
    assert cfg.grad_currency is True
    assert cfg.enable_confidence is True
    assert cfg.confidence_mode == "twod"
    # pinned to the HARD cliff so it stays the regression control / A-B baseline
    assert cfg.settled_mode == "hard"


def test_currency_2dsoft_variant_uses_softened_sigmoid_cliff():
    cfg = make_config("currency-2dsoft")
    assert cfg.grad_currency is True
    assert cfg.confidence_mode == "twod"
    assert cfg.settled_mode == "sigmoid"
    assert cfg.conf_k == 3.0


def test_currency_grace_extends_currency():
    cfg = make_config("currency-grace")
    assert cfg.grad_currency is True
    assert cfg.t_grace == 1000
    assert cfg.grow_bar_frac == 2.0


def test_core_variant_is_plain_sgd():
    cfg = make_config("core")
    assert cfg.grad_currency is False
    assert cfg.enable_eligibility is False
    assert cfg.enable_prune is False
    assert cfg.enable_grow is False


def test_make_config_returns_fresh_instances():
    a = make_config("currency")
    b = make_config("currency")
    assert a is not b  # mutating one must not affect another run


def test_unknown_variant_raises():
    with pytest.raises(KeyError):
        make_config("does-not-exist")


def test_registry_lists_all_named_variants():
    assert set(VARIANTS) >= {"legacy-full", "currency", "currency-grace", "core"}


def test_suitespec_defaults():
    spec = SuiteSpec()
    assert spec.variants == ("currency", "legacy-full")
    assert spec.seeds == 5
    assert spec.dataset == "spirals"
    assert spec.steps == 30000
    assert spec.shift_steps == 0
    assert spec.baseline == "currency"   # promoted: softened-cliff currency is the reference
    assert spec.record_every == 200
    assert spec.test_seed_offset == 10000


def test_suitespec_seed_list_is_deterministic_range():
    spec = SuiteSpec(seeds=3)
    assert list(spec.seed_list()) == [0, 1, 2]


def test_suitespec_continual_defaults():
    spec = SuiteSpec()
    assert spec.regime == "single"      # existing single-task + label-swap path
    assert spec.steps_a == 15000
    assert spec.steps_b == 15000
    assert spec.steps_ab == 10000
    # concentric geometry: gentler spirals so the union is learnable, an inner
    # annular spiral (A) and a disjoint outer one (B), both origin-centred.
    assert spec.continual_turns == 0.6
    assert (spec.inner_r_lo, spec.inner_r_hi) == (0.15, 0.55)
    assert (spec.outer_r_lo, spec.outer_r_hi) == (0.65, 1.05)


def test_suitespec_continual_regime_opt_in():
    spec = SuiteSpec(regime="continual", steps_a=100, steps_b=100, steps_ab=50)
    assert spec.regime == "continual"
    assert spec.steps_a == 100
