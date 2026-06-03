"""Tests for the settledness detector (sprout/sleep.py)."""

from __future__ import annotations

from sprout.sleep import SettlednessDetector


def test_not_settled_before_warmup():
    # even a perfectly flat (plateaued) loss must not settle under the warmup.
    d = SettlednessDetector(beta=0.5, tol=0.01, patience=2, warmup=100)
    assert not any(d.update(1.0, step) for step in range(50))


def test_settles_after_patience_on_plateau():
    # beta=1.0 => EMA tracks raw loss exactly, so the patience logic is tested
    # without EMA-lag confounds (real runs use a small beta to smooth noise).
    d = SettlednessDetector(beta=1.0, tol=0.01, patience=3, warmup=0)
    # a strictly improving loss never settles
    assert not d.update(1.0, 0)
    assert not d.update(0.5, 1)
    assert not d.update(0.25, 2)
    # now flat: improvements stop -> after `patience` flat steps, settled
    flags = [d.update(0.25, s) for s in range(3, 9)]
    assert flags[0] is False   # not immediately
    assert flags[-1] is True   # eventually


def test_new_improvement_resets_since_improve():
    d = SettlednessDetector(beta=1.0, tol=0.01, patience=2, warmup=0)
    d.update(1.0, 0)
    d.update(1.0, 1)
    d.update(1.0, 2)                       # plateau building
    assert d.update(0.5, 3) is False       # a big improvement resets the counter
    assert d.update(0.5, 4) is False       # rebuilding patience from scratch


def test_reset_requires_fresh_plateau():
    d = SettlednessDetector(beta=1.0, tol=0.01, patience=2, warmup=0)
    for s in range(5):
        d.update(1.0, s)
    assert d.update(1.0, 5) is True        # settled after a long plateau
    d.reset()
    assert d.update(1.0, 6) is False       # must re-plateau before the next burst


def test_first_loss_seeds_the_ema():
    # the EMA seeds on the first loss (no zero-init transient that looks like a
    # huge improvement); with beta=1.0 the EMA tracks the raw loss exactly.
    d = SettlednessDetector(beta=1.0, tol=0.01, patience=1, warmup=0)
    assert d.update(0.42, 0) is False
    assert d.loss_ema == 0.42
