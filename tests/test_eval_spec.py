"""Tests for the variant registry and suite specification."""

from __future__ import annotations

import pytest

from evals.spec import VARIANTS, make_config, SuiteSpec
from sprout.train import Config


def test_currency_variant_uses_gradient_currency():
    cfg = make_config("currency")
    assert cfg.grad_currency is True
    assert cfg.enable_confidence and cfg.enable_prune and cfg.enable_grow
    # the promoted baseline: default currency now uses the calibrated 2D
    # confidence with the softened (sigmoid) settled cliff
    assert cfg.confidence_mode == "twod"
    assert cfg.settled_mode == "sigmoid"
    # ...and the selective hiring bar (B1 promotion), not the prior eager 1.5
    assert cfg.grow_bar_frac == 3.0


def test_currency_eager_variant_preserves_the_prior_growth_bar():
    # the prior eager growth default (grow_bar_frac=1.5) is kept as an explicit
    # variant for comparison even though 3.0 is now the baseline
    cfg = make_config("currency-eager")
    assert cfg.grad_currency is True
    assert cfg.grow_bar_frac == 1.5
    assert cfg.ghost_meter is False


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


def test_b1_growbar_variants_vary_only_the_grow_bar():
    # B1: grow_bar_frac swept, t_grace pinned at the baseline (clean isolation)
    for name, gb in [("currency-gb2", 2.0), ("currency-gb3", 3.0)]:
        cfg = make_config(name)
        assert cfg.grow_bar_frac == gb
        assert cfg.t_grace == 200            # unchanged from baseline
        assert cfg.ghost_meter is False


def test_c1_grace_variants_vary_only_the_grace_window():
    # C1: t_grace swept, grow_bar_frac pinned at the baseline (clean isolation)
    for name, tg in [("currency-grace500", 500), ("currency-grace1k", 1000),
                     ("currency-grace2k", 2000)]:
        cfg = make_config(name)
        assert cfg.t_grace == tg
        assert cfg.grow_bar_frac == 1.5      # unchanged from baseline
        assert cfg.ghost_meter is False


def test_a2_ghost_variants_enable_the_ghost_meter():
    cfg = make_config("currency-ghost")
    assert cfg.ghost_meter is True
    assert cfg.beta_ghost == 0.8
    assert cfg.grow_bar_frac == 1.5 and cfg.t_grace == 200   # only the meter differs
    strong = make_config("currency-ghost-strong")
    assert strong.ghost_meter is True
    assert strong.beta_ghost == 0.9


def test_gb3_ghost_combo_stacks_both_winning_levers():
    # the combined variant turns on BOTH the B1 (higher bar) and A2 (ghost meter)
    # levers at once — nothing else changes from the currency baseline
    cfg = make_config("currency-gb3-ghost")
    assert cfg.grow_bar_frac == 3.0       # B1
    assert cfg.ghost_meter is True        # A2
    assert cfg.beta_ghost == 0.8
    assert cfg.t_grace == 200             # baseline (C1 not stacked)


def test_core_variant_is_plain_sgd():
    cfg = make_config("core")
    assert cfg.enable_confidence is False
    assert cfg.enable_prune is False
    assert cfg.enable_grow is False


def test_static_matched_variant_is_plain_sgd_at_baseline_budget():
    # the synapse-count-matched control: a STATIC sparse net (all plasticity off)
    # that inherits the suite --density, so it starts from the same random graph
    # as the currency baseline (same seed) and never rewires. Isolates the effect
    # of adaptive rewiring at an equal wire budget. Config-identical to `core`.
    cfg = make_config("static-matched")
    assert cfg.init_density is None          # uses the suite density, matching baseline
    assert cfg.eta_base == 0.02
    assert cfg.enable_confidence is False
    assert cfg.enable_prune is False
    assert cfg.enable_grow is False


def test_fully_connected_variant_is_a_dense_static_mlp():
    # the "fully connected" comparison arm: a dense, all-to-all graph
    # (init_density=1.0) trained with plain single-sample SGD. Every plasticity
    # mechanism is OFF, so the topology never changes — the brute-force control
    # against the sparse, self-rewiring `currency` baseline.
    cfg = make_config("fully-connected")
    assert cfg.init_density == 1.0
    assert cfg.eta_base == 0.02
    assert cfg.enable_confidence is False
    assert cfg.enable_prune is False
    assert cfg.enable_grow is False


def test_size_sweep_variants_are_currency_gb3_at_varied_widths():
    # the neuron-count sweep: the promoted currency-gb3 config held fixed, only
    # the hidden-layer width varies (uniform 3-hidden-layer topology). init_density
    # stays None so each arm uses the suite's sparse density — only neuron count,
    # not connectivity regime, changes across the sweep.
    expected = {
        "size-w4": (2, 4, 4, 4, 2),
        "size-w6": (2, 6, 6, 6, 2),
        "size-w10": (2, 10, 10, 10, 2),
        "size-w16": (2, 16, 16, 16, 2),
        "size-w24": (2, 24, 24, 24, 2),
    }
    for name, layers in expected.items():
        cfg = make_config(name)
        assert cfg.grad_currency is True
        assert cfg.enable_confidence and cfg.enable_prune and cfg.enable_grow
        assert cfg.confidence_mode == "twod" and cfg.settled_mode == "sigmoid"
        assert cfg.grow_bar_frac == 3.0          # the promoted selective bar
        assert cfg.init_layers == layers
        assert cfg.init_density is None          # sparse, suite-density wiring


def test_sleep_variant_is_currency_plus_the_promoted_default():
    # `sleep` = the no-sleep `currency` baseline + the PROMOTED default sleep
    # (floor 1.0, no cap). currency pins sleep off; sleep inherits the new default.
    base = make_config("currency")
    s = make_config("sleep")
    assert base.enable_sleep is False              # pinned baseline
    assert s.enable_sleep is True
    assert s.sleep_prune_floor == 1.0 and s.sleep_max_prune is None  # the default
    assert s.grad_currency and s.enable_prune and s.enable_grow
    assert s.confidence_mode == base.confidence_mode
    assert s.settled_mode == base.settled_mode
    assert s.grow_bar_frac == base.grow_bar_frac


def test_sleep_deep_is_a_bounded_floor3_variant():
    # sleep-deep is a historical bounded-aggressive arm (floor 3, capped 20), kept
    # for comparison — distinct from the promoted no-cap default `sleep`.
    deep = make_config("sleep-deep")
    assert deep.enable_sleep is True
    assert deep.sleep_prune_floor == 3.0
    assert deep.sleep_max_prune == 20              # bounded (unlike the no-cap default)
    assert deep.sleep_warmup == 2000 and deep.sleep_patience == 800


def test_sleep_prune_sweep_scales_aggressiveness_monotonically():
    # the deeper-prune sweep: floor AND per-burst cap both rise together so each
    # arm consolidates strictly harder than the last (maps the accuracy tail-off).
    sweep = ["sleep-f2", "sleep-f3", "sleep-f4", "sleep-f5", "sleep-f6"]
    floors = [make_config(n).sleep_prune_floor for n in sweep]
    caps = [make_config(n).sleep_max_prune for n in sweep]
    assert floors == sorted(floors) and len(set(floors)) == 5   # strictly increasing
    assert caps == sorted(caps) and len(set(caps)) == 5
    for n in sweep:
        cfg = make_config(n)
        assert cfg.enable_sleep is True
        assert cfg.sleep_warmup == 2000 and cfg.sleep_patience == 800
        assert cfg.grad_currency and cfg.enable_prune and cfg.enable_grow


def test_sleep_nocap_sweep_varies_floor_with_no_effective_cap():
    # the no-cap sweep: same fire-often frame, but the per-burst cap far exceeds
    # the wire count so the FLOOR is the sole lever — each burst removes EVERY
    # eligible (below-floor, non-orphan) wire in one shot. Isolates floor depth.
    sweep = {"sleep-nc2": 2.0, "sleep-nc3": 3.0, "sleep-nc4": 4.0,
             "sleep-nc5": 5.0, "sleep-nc6": 6.0}
    for name, floor in sweep.items():
        cfg = make_config(name)
        assert cfg.enable_sleep is True
        assert cfg.sleep_prune_floor == floor
        assert cfg.sleep_max_prune >= 100000          # effectively uncapped
        assert cfg.sleep_warmup == 2000 and cfg.sleep_patience == 800
        assert cfg.grad_currency and cfg.enable_prune and cfg.enable_grow


def test_sleep_lowfloor_nocap_sweep_varies_floor_below_one():
    # the LOW-floor no-cap sweep: uncapped bursts, but floors 0.2-1.0 — all BELOW
    # the median wire utility (~1.7) and around the default wake floor (0.5), so
    # each burst removes only the genuinely-weak tail. Probes where uncapped
    # pruning starts to bite (the high-floor no-cap sweep collapsed because 2-6
    # were above the median). Names use floor x 10: lo2 = 0.2 ... lo10 = 1.0.
    sweep = {"sleep-lo2": 0.2, "sleep-lo4": 0.4, "sleep-lo6": 0.6,
             "sleep-lo8": 0.8, "sleep-lo10": 1.0}
    for name, floor in sweep.items():
        cfg = make_config(name)
        assert cfg.enable_sleep is True
        assert cfg.sleep_prune_floor == floor
        assert cfg.sleep_max_prune >= 100000          # no cap
        assert cfg.sleep_warmup == 2000 and cfg.sleep_patience == 800
        assert cfg.grad_currency and cfg.enable_prune and cfg.enable_grow


def test_sleep_midfloor_nocap_arms_fill_one_to_two():
    # fills the 1.0->2.0 gap in the no-cap floor sweep (between the safe lo10=1.0
    # and the collapsing nc2=2.0), so the full 0->2 curve resolves the cliff.
    sweep = {"sleep-lo12": 1.2, "sleep-lo14": 1.4,
             "sleep-lo16": 1.6, "sleep-lo18": 1.8}
    for name, floor in sweep.items():
        cfg = make_config(name)
        assert cfg.enable_sleep is True
        assert cfg.sleep_prune_floor == floor
        assert cfg.sleep_max_prune >= 100000          # no cap
        assert cfg.sleep_warmup == 2000 and cfg.sleep_patience == 800
        assert cfg.grad_currency and cfg.enable_prune and cfg.enable_grow


def test_make_config_returns_fresh_instances():
    a = make_config("currency")
    b = make_config("currency")
    assert a is not b  # mutating one must not affect another run


def test_unknown_variant_raises():
    with pytest.raises(KeyError):
        make_config("does-not-exist")


def test_registry_lists_all_named_variants():
    assert set(VARIANTS) >= {"currency", "sleep", "currency-grace", "core"}


def test_suitespec_defaults():
    spec = SuiteSpec()
    assert spec.variants == ("phasic-startle-k4", "phasic-startle")
    assert spec.seeds == 5
    assert spec.dataset == "spirals"
    # promoted defaults: the width sweep made w16 the sweet spot, and the
    # single-task horizon is now 15k (w16 converges well within it).
    assert spec.steps == 15000
    assert spec.layers == (2, 16, 16, 16, 2)
    assert spec.shift_steps == 0
    assert spec.baseline == "phasic-startle-k4"   # promoted sparse/efficient reference
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


def test_currency_bounded_is_currency_plus_demand_k():
    from evals.spec import make_config
    base = make_config("currency")
    bounded = make_config("currency-bounded")
    assert base.grow_demand_k is None
    assert isinstance(bounded.grow_demand_k, int) and bounded.grow_demand_k > 0
    # identical to currency except for the demand bound
    assert bounded.grad_currency and bounded.enable_grow and bounded.enable_prune
    assert bounded.grow_bar_frac == base.grow_bar_frac


def test_phasic_startle_research_variants_are_single_knob_extensions():
    base = make_config("phasic-startle")
    k4 = make_config("phasic-startle-k4")
    aroused = make_config("phasic-startle-aroused")
    both = make_config("phasic-startle-aroused-k4")

    assert base.phasic_structure and base.startle
    assert base.grow_demand_k is None
    assert base.arousal_steps == 0

    assert k4.phasic_structure and k4.startle
    assert k4.grow_demand_k == 4
    assert k4.arousal_steps == 0

    assert aroused.phasic_structure and aroused.startle
    assert aroused.grow_demand_k is None
    assert aroused.arousal_steps == 1000

    assert both.phasic_structure and both.startle
    assert both.grow_demand_k == 4
    assert both.arousal_steps == 1000


def test_make_config_pins_historical_full_scan_variants():
    # Raw Config defaults to k4 now; old named comparators still mean full-scan
    # unless the variant name explicitly says k4/bounded.
    for name in ("currency", "sleep", "phasic", "phasic-startle", "size-w16"):
        assert make_config(name).grow_demand_k is None


def test_bounded_size_sweep_arms_match_widths_with_demand_k():
    from evals.spec import make_config
    widths = {"size-w4-k4": (2, 4, 4, 4, 2), "size-w6-k4": (2, 6, 6, 6, 2),
              "size-w10-k4": (2, 10, 10, 10, 2), "size-w16-k4": (2, 16, 16, 16, 2),
              "size-w24-k4": (2, 24, 24, 24, 2)}
    for name, layers in widths.items():
        cfg = make_config(name)
        assert cfg.init_layers == layers
        assert cfg.grow_demand_k == 4
        assert cfg.grad_currency and cfg.enable_grow and cfg.enable_prune


def test_compute_efficiency_probe_variants_are_single_knobs():
    base = make_config("phasic-startle-k4")
    assert make_config("phasic-startle-k4-lazy").lazy_meters is True
    assert make_config("eff-density30").init_density == 0.30
    assert make_config("eff-density50").init_density == 0.50
    assert make_config("eff-w12").init_layers == (2, 12, 12, 12, 2)
    assert make_config("eff-w20").init_layers == (2, 20, 20, 20, 2)
    assert make_config("eff-floor08").sleep_prune_floor == 0.8
    assert make_config("eff-floor12").sleep_prune_floor == 1.2
    assert make_config("eff-floor15").sleep_prune_floor == 1.5
    assert make_config("eff-wta4").activation_top_k == 4
    assert make_config("eff-wta6").activation_top_k == 6
    assert make_config("eff-wta8").activation_top_k == 8

    for name in ("phasic-startle-k4-lazy", "eff-density30", "eff-density50",
                 "eff-w12", "eff-w20", "eff-floor08", "eff-floor12",
                 "eff-floor15", "eff-wta4", "eff-wta6", "eff-wta8"):
        cfg = make_config(name)
        assert cfg.grow_demand_k == base.grow_demand_k == 4
        assert cfg.phasic_structure and cfg.startle
        assert cfg.enable_confidence and cfg.enable_prune and cfg.enable_grow


def test_digit_width_sweep_matched_edge_budget():
    """The width-sweep arms all start within ~8% of the same edge budget, with
    the dense arm static and the wide arms self-rewiring (phasic-startle-k4)."""
    from sprout.network import build_graph
    arms = {
        "digits-w16-dense":  ((64, 16, 10), 1.0, False),
        "digits-w32-sparse": ((64, 32, 10), 0.5, True),
        "digits-w64-sparse": ((64, 64, 10), 0.25, True),
        "digits-w128-sparse": ((64, 128, 10), 0.125, True),
    }
    counts = []
    for name, (layers, density, sparse) in arms.items():
        cfg = make_config(name)
        assert cfg.init_layers == layers
        assert cfg.init_density == density
        if sparse:
            assert cfg.phasic_structure and cfg.startle and cfg.grow_demand_k == 4
            assert cfg.enable_confidence and cfg.enable_prune and cfg.enable_grow
        else:
            assert cfg.init_density == 1.0 and not cfg.enable_grow
        counts.append(len(build_graph(list(layers), density=density, seed=0).synapses))
    assert max(counts) / min(counts) < 1.08    # matched compute budget


def test_digit_w128_kscale_variants():
    """w128 k-scaling probe: same wide net + budget, larger grow_demand_k."""
    assert make_config("digits-w128-k16").grow_demand_k == 16
    assert make_config("digits-w128-k32").grow_demand_k == 32
    for name in ("digits-w128-k16", "digits-w128-k32"):
        cfg = make_config(name)
        assert cfg.init_layers == (64, 128, 10) and cfg.init_density == 0.125
        assert cfg.phasic_structure and cfg.startle and cfg.enable_grow


def test_digit_budget_floor_sweep_variants():
    """Narrow the w32-sparse winner at fixed density 0.5 to probe the edge floor:
    fewer hidden neurons -> fewer edges, fan-in into hidden held ~32."""
    from sprout.network import build_graph
    arms = [("digits-w8-sparse", 8), ("digits-w12-sparse", 12),
            ("digits-w16-sparse", 16), ("digits-w24-sparse", 24)]
    prev = 0
    for name, w in arms:
        cfg = make_config(name)
        assert cfg.init_layers == (64, w, 10) and cfg.init_density == 0.5
        assert cfg.grow_demand_k == 4 and cfg.phasic_structure and cfg.startle
        n = len(build_graph([64, w, 10], density=0.5, seed=0).synapses)
        assert n > prev    # monotonically more edges with width
        prev = n


def test_mnist_width_sweep_matched_edge_budget():
    """Downsampled-MNIST (196-in) width sweep: dense w16 vs wide sparse arms,
    all within ~8% of the same ~3296-edge budget."""
    from sprout.network import build_graph
    arms = {
        "mnist-w16-dense":  ((196, 16, 10), 1.0, False),
        "mnist-w32-sparse": ((196, 32, 10), 0.5, True),
        "mnist-w64-sparse": ((196, 64, 10), 0.25, True),
        "mnist-w128-sparse": ((196, 128, 10), 0.125, True),
    }
    counts = []
    for name, (layers, density, sparse) in arms.items():
        cfg = make_config(name)
        assert cfg.init_layers == layers and cfg.init_density == density
        if sparse:
            assert cfg.phasic_structure and cfg.startle and cfg.grow_demand_k == 4
        else:
            assert cfg.init_density == 1.0 and not cfg.enable_grow
        counts.append(len(build_graph(list(layers), density=density, seed=0).synapses))
    assert max(counts) / min(counts) < 1.08


def test_mnist_widen_budget_variants():
    """2x-budget wide arms (~6592 edges vs the 3296 matched budget), w64
    restored to w32's healthy fan-in (98)."""
    from sprout.network import build_graph
    for name, layers, density in [("mnist-w64-b2", (196, 64, 10), 0.5),
                                  ("mnist-w128-b2", (196, 128, 10), 0.25)]:
        cfg = make_config(name)
        assert cfg.init_layers == layers and cfg.init_density == density
        assert cfg.phasic_structure and cfg.startle and cfg.grow_demand_k == 4
        n = len(build_graph(list(layers), density=density, seed=0).synapses)
        assert 6000 < n < 7200          # ~2x the 3296 matched budget
