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
    # the current default architecture: gradient-as-currency. Confidence is the
    # calibrated 2D (importance x settledness) rule with the softened sigmoid
    # cliff, and growth uses the selective hiring bar (grow_bar_frac=3.0) — both
    # inherited from Config defaults. This is the promoted BASELINE.
    "currency": lambda: Config(
        eta_base=0.02, grad_currency=True, enable_confidence=True,
        enable_prune=True, enable_grow=True,
        gamma_dec=0.001, t_struct=200,
    ),
    # currency with the PRIOR eager growth bar (grow_bar_frac=1.5), kept for
    # comparison now that the selective 3.0 bar is the default. The eager bar grew
    # ~2x as many wires and drove the grow<->prune oscillation (docs/eval-runs/
    # b1-growbar-sweep). Mirror of currency-tugofwar for the confidence rule.
    "currency-eager": lambda: Config(
        eta_base=0.02, grad_currency=True, enable_confidence=True,
        enable_prune=True, enable_grow=True,
        gamma_dec=0.001, t_struct=200, grow_bar_frac=1.5,
    ),
    # currency with the prior tug-of-war confidence rule (calm+consistent earn/
    # lose), kept for comparison now that 2D+softened-cliff is the default.
    "currency-tugofwar": lambda: Config(
        eta_base=0.02, grad_currency=True, enable_confidence=True,
        enable_prune=True, enable_grow=True,
        gamma_dec=0.001, t_struct=200, confidence_mode="tugofwar",
    ),
    # currency with the 2D (importance x settledness) confidence rule, HARD cliff
    # — the original calibration redesign; the A/B baseline for the softened cliff.
    "currency-2dconf": lambda: Config(
        eta_base=0.02, grad_currency=True, enable_confidence=True,
        enable_prune=True, enable_grow=True,
        gamma_dec=0.001, t_struct=200, confidence_mode="twod", settled_mode="hard",
    ),
    # 2D confidence with the SOFTENED settled cliff (sigmoid): a contested
    # load-bearer keeps some consolidation instead of collapsing to zero. NOTE:
    # now config-identical to the default "currency"; kept as an explicit alias
    # for reproducing the 2dsoft-vs-2dconf published runs.
    "currency-2dsoft": lambda: Config(
        eta_base=0.02, grad_currency=True, enable_confidence=True,
        enable_prune=True, enable_grow=True,
        gamma_dec=0.001, t_struct=200, confidence_mode="twod",
        settled_mode="sigmoid", conf_k=3.0,
    ),
    # currency with a longer grace + higher grow bar (the grow_budget replacement)
    "currency-grace": lambda: Config(
        eta_base=0.02, grad_currency=True, enable_confidence=True,
        enable_prune=True, enable_grow=True,
        gamma_dec=0.001, t_struct=200, t_grace=1000, grow_bar_frac=2.0,
    ),

    # === anti-oscillation experiments ========================================
    # CONCLUDED: the B1 grow-bar sweep promoted grow_bar_frac=3.0 to the Config
    # default (so `currency` above now IS the selective bar). The C1 (grace) and
    # A2 (ghost) variants are kept pinned to grow_bar_frac=1.5 — the PRIOR eager
    # default they were measured against — so each stays a faithful single-knob
    # sweep and reproduces its published folder (docs/eval-runs/{c1-grace-sweep,
    # a2-ghost-meter}). The confidence comparators above instead track the current
    # baseline. See docs/eval-runs/{b1-growbar-sweep,gb3-ghost-combo}.
    #
    # B1 — raise the hiring bar (Schmitt-trigger gap): only robustly-wanted wires
    # are born, so once born they clear the prune floor comfortably. gb3 is now
    # config-identical to `currency` (kept as an explicit alias for the sweep);
    # gb2 is the intermediate point.
    "currency-gb2": lambda: Config(
        eta_base=0.02, grad_currency=True, enable_confidence=True,
        enable_prune=True, enable_grow=True,
        gamma_dec=0.001, t_struct=200, grow_bar_frac=2.0,
    ),
    "currency-gb3": lambda: Config(
        eta_base=0.02, grad_currency=True, enable_confidence=True,
        enable_prune=True, enable_grow=True,
        gamma_dec=0.001, t_struct=200, grow_bar_frac=3.0,
    ),
    # C1 — longer probation (the LOSS: cut max_regrow only by postponing pruning,
    # leaving a denser net with more freeloaders; oscillation_frac unmoved).
    # Pinned to the prior eager bar (grow_bar_frac=1.5) so t_grace is isolated.
    "currency-grace500": lambda: Config(
        eta_base=0.02, grad_currency=True, enable_confidence=True,
        enable_prune=True, enable_grow=True,
        gamma_dec=0.001, t_struct=200, t_grace=500, grow_bar_frac=1.5,
    ),
    "currency-grace1k": lambda: Config(
        eta_base=0.02, grad_currency=True, enable_confidence=True,
        enable_prune=True, enable_grow=True,
        gamma_dec=0.001, t_struct=200, t_grace=1000, grow_bar_frac=1.5,
    ),
    "currency-grace2k": lambda: Config(
        eta_base=0.02, grad_currency=True, enable_confidence=True,
        enable_prune=True, enable_grow=True,
        gamma_dec=0.001, t_struct=200, t_grace=2000, grow_bar_frac=1.5,
    ),
    # A2 — ghost-gradient meter: grow on a persistent EMA of the virtual gradient
    # so a just-pruned wire must re-earn growth over several cycles (soft
    # refractory) instead of being re-requested on the next noisy batch spike. Cut
    # max_regrow strongly but not oscillation_frac; partly redundant once the bar
    # is high (gb3-ghost-combo). Pinned to the prior eager bar to isolate the meter.
    "currency-ghost": lambda: Config(
        eta_base=0.02, grad_currency=True, enable_confidence=True,
        enable_prune=True, enable_grow=True,
        gamma_dec=0.001, t_struct=200, ghost_meter=True, beta_ghost=0.8,
        grow_bar_frac=1.5,
    ),
    "currency-ghost-strong": lambda: Config(
        eta_base=0.02, grad_currency=True, enable_confidence=True,
        enable_prune=True, enable_grow=True,
        gamma_dec=0.001, t_struct=200, ghost_meter=True, beta_ghost=0.9,
        grow_bar_frac=1.5,
    ),
    # B1 + A2 stacked: pickier hiring (grow_bar_frac=3.0, shrinks how MANY wires
    # thrash) + sustained-signal growth (ghost meter, cuts how HARD the worst one
    # thrashes). The two levers fixed different halves of the oscillation in the
    # sweeps; this tests whether the wins stack (and whether the small post-shift
    # recovery dip compounds — hence the paired shift guardrail run).
    "currency-gb3-ghost": lambda: Config(
        eta_base=0.02, grad_currency=True, enable_confidence=True,
        enable_prune=True, enable_grow=True,
        gamma_dec=0.001, t_struct=200,
        grow_bar_frac=3.0, ghost_meter=True, beta_ghost=0.8,
    ),
    # === neuron-count (width) sweep =========================================
    # The promoted currency-gb3 config held FIXED while only the hidden-layer
    # width varies (uniform 3-hidden-layer topology, input/output pinned by the
    # 2D spirals task). init_density stays None so every arm uses the suite's
    # sparse density — only neuron count, not connectivity regime, changes. The
    # sweep asks how network size trades off learning speed, accuracy, and neuron
    # utilisation ("average neuron value"). The sweep's own baseline was size-w10
    # (≈ the THEN-standard (2,10,10,8,2) net); its result promoted w16 to the new
    # default topology (see SuiteSpec.layers).
    "size-w4": lambda: Config(
        eta_base=0.02, grad_currency=True, enable_confidence=True,
        enable_prune=True, enable_grow=True,
        gamma_dec=0.001, t_struct=200, init_layers=(2, 4, 4, 4, 2)),
    "size-w6": lambda: Config(
        eta_base=0.02, grad_currency=True, enable_confidence=True,
        enable_prune=True, enable_grow=True,
        gamma_dec=0.001, t_struct=200, init_layers=(2, 6, 6, 6, 2)),
    "size-w10": lambda: Config(
        eta_base=0.02, grad_currency=True, enable_confidence=True,
        enable_prune=True, enable_grow=True,
        gamma_dec=0.001, t_struct=200, init_layers=(2, 10, 10, 10, 2)),
    "size-w16": lambda: Config(
        eta_base=0.02, grad_currency=True, enable_confidence=True,
        enable_prune=True, enable_grow=True,
        gamma_dec=0.001, t_struct=200, init_layers=(2, 16, 16, 16, 2)),
    "size-w24": lambda: Config(
        eta_base=0.02, grad_currency=True, enable_confidence=True,
        enable_prune=True, enable_grow=True,
        gamma_dec=0.001, t_struct=200, init_layers=(2, 24, 24, 24, 2)),

    # plain sparse SGD, all plasticity off (a floor reference)
    "core": lambda: Config(eta_base=0.02),
    # the synapse-count-matched control for the fully-connected comparison: a
    # STATIC sparse net (all plasticity off) that inherits the suite --density, so
    # it starts from the SAME random graph as the currency baseline (same seed)
    # and never rewires. Plain single-sample SGD. Isolates "does adaptive rewiring
    # earn its keep at an equal wire budget?" (vs `currency`) and "was the dense
    # net's speed just extra wires?" (vs `fully-connected`). Config-identical to
    # `core`; named for its role in the matched-synapse-count experiment.
    "static-matched": lambda: Config(eta_base=0.02),
    # the "fully connected" comparison arm: a dense, all-to-all MLP
    # (init_density=1.0) trained with plain single-sample SGD — every plasticity
    # mechanism off, so the topology is fixed. The brute-force control against the
    # sparse, self-rewiring `currency` baseline: does self-organised sparsity
    # learn as fast, and adapt to a second task as quickly, on far fewer synapses?
    "fully-connected": lambda: Config(eta_base=0.02, init_density=1.0),
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
    steps: int = 15000
    shift_steps: int = 0          # > 0 enables a mid-training label-swap phase
    record_every: int = 200
    baseline: str = "currency"     # softened-cliff 2D-confidence currency = the reference
    # promoted to w16 (uniform 16-wide hidden layers): the neuron-width sweep
    # found it the efficiency sweet spot — near-top accuracy and ~1.8x faster
    # convergence than the old (2,10,10,8,2) at ~2x the wires, with the fewest
    # freeloaders/idle units (docs/eval-runs/neuron-width-sweep).
    layers: tuple[int, ...] = (2, 16, 16, 16, 2)
    density: float = 0.4
    n_points: int = 600
    turns: float = 1.0
    noise: float = 0.10
    test_seed_offset: int = 10000  # held-out test set drawn at seed + this

    # continual-learning (forgetting) regime: two CONCENTRIC spirals, A->B->A+B.
    # Both tasks are origin-centred (zero-mean => learnable by the tiny net) but
    # disjoint by radius (jointly valid): inner annular spiral = A, outer = B.
    # regime="single" keeps the existing single-task + optional label-swap path.
    regime: str = "single"         # "single" | "continual"
    steps_a: int = 15000           # phase A: learn the inner spiral
    steps_b: int = 15000           # phase B: learn the outer spiral only (A erodes)
    steps_ab: int = 10000          # phase A+B: interleaved consolidation
    continual_turns: float = 0.6   # gentler spirals so the 4-arm union is learnable
    inner_r_lo: float = 0.15       # task A: inner annular spiral radial band
    inner_r_hi: float = 0.55
    outer_r_lo: float = 0.65       # task B: disjoint outer ring (gap at ~0.6)
    outer_r_hi: float = 1.05

    def seed_list(self) -> range:
        return range(self.seeds)
