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
    assert spec.baseline == "legacy-full"
    assert spec.record_every == 200
    assert spec.test_seed_offset == 10000


def test_suitespec_seed_list_is_deterministic_range():
    spec = SuiteSpec(seeds=3)
    assert list(spec.seed_list()) == [0, 1, 2]
