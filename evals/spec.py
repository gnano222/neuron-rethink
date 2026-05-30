"""Variant registry and suite specification for the evaluation harness.

A *variant* is a named training configuration. The registry bakes in the tuned
spirals hyperparameters each architecture needs (lifted from the proven
``compare.py`` / ``validate.py`` configs), so the harness compares architectures
on their *intended* settings rather than the bare ``run.PRESETS`` flag combos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sprout.train import Config


# name -> factory returning a FRESH Config (never share mutable Config instances
# across runs; each (variant, seed) job mutates its own network/config state).
VARIANTS: dict[str, Callable[[], Config]] = {
    # the README's tuned v1 eligibility system (the baseline to beat)
    "legacy-full": lambda: Config(
        eta_base=0.02, enable_eligibility=True, enable_confidence=True,
        enable_prune=True, enable_grow=True,
        theta_prune=0.001, prune_warmup=6000,
    ),
    # the current default architecture: gradient-as-currency, out of the box
    "currency": lambda: Config(
        eta_base=0.02, grad_currency=True, enable_confidence=True,
        enable_prune=True, enable_grow=True,
        gamma_dec=0.001, t_struct=200,
    ),
    # currency with a longer grace + higher grow bar (the grow_budget replacement)
    "currency-grace": lambda: Config(
        eta_base=0.02, grad_currency=True, enable_confidence=True,
        enable_prune=True, enable_grow=True,
        gamma_dec=0.001, t_struct=200, t_grace=1000, grow_bar_frac=2.0,
    ),
    # plain sparse SGD, all plasticity off (a floor reference)
    "core": lambda: Config(eta_base=0.02),
}


def make_config(name: str) -> Config:
    """Return a fresh :class:`Config` for the named variant."""
    if name not in VARIANTS:
        raise KeyError(
            f"unknown variant {name!r}; known: {', '.join(sorted(VARIANTS))}")
    return VARIANTS[name]()


@dataclass
class SuiteSpec:
    """Everything needed to run and aggregate one comparison."""

    variants: tuple[str, ...] = ("currency", "legacy-full")
    seeds: int = 5
    dataset: str = "spirals"
    steps: int = 30000
    shift_steps: int = 0          # > 0 enables a mid-training label-swap phase
    record_every: int = 200
    baseline: str = "legacy-full"
    layers: tuple[int, ...] = (2, 10, 10, 8, 2)
    density: float = 0.4
    n_points: int = 600
    turns: float = 1.0
    noise: float = 0.10
    test_seed_offset: int = 10000  # held-out test set drawn at seed + this

    def seed_list(self) -> range:
        return range(self.seeds)
