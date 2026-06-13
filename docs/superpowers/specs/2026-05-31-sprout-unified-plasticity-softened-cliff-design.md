# SPROUT unified plasticity (one state, two lenses) + softened settled cliff — design

**Date:** 2026-05-31
**Branch:** sprout-v1
**Status:** approved, ready for implementation

## Goal

Make confidence and prune-utility legibly **two lenses on one shared per-wire
state**, and **soften the `settled` cliff** so the high-demand tail stops
collapsing to exactly zero confidence.

Two motivations, from inspecting the last `currency-2dconf` run:

1. **They feel like two overlapping systems.** `update_confidence_2d`,
   `prune_currency`, and `metrics.synapse_utilities` each independently recompute
   `w̄` / `M̄` (the metric even uses *fresh* demand instead of the meter). They
   are all functions of the same two signals — weight and gradient — but that
   isn't expressed anywhere.
2. **The tail.** In the confidence-vs-utility scatter there is a band of wires
   with **high utility, low/zero confidence**. Root cause: utility counts demand
   with a **+** sign (`U = ℓ + λd`) while confidence's `settled = max(1 − d, 0)`
   counts it with a **−** sign and **slams to exactly 0 for every wire with
   above-average demand** (`d ≥ 1`), regardless of how much load it carries. A
   load-bearing wire that gets briefly contested loses *all* its consolidation in
   one step. This both wastes consolidation and drags `conf_utility_corr` down
   (the metric reads it as miscalibration even though "keep" and "freeze" are
   genuinely different questions).

## Key insight (why we do NOT collapse to one scalar)

Weight `w` (load now) and gradient `M` (demand / expected future value) are two
axes. A newborn wire `(w≈0, M high)` and a frozen load-bearer `(w high, M≈0)`
have **similar "worth keeping"** but need **opposite plasticity** (newborn: stay
plastic; load-bearer: rigid). One scalar cannot drive both pruning and the
learning-rate gate, because the gradient term must flip sign between them. The
genuinely-single thing is **the 2D state itself**; the two mechanisms are two
projections of it. Approved direction: **one state, two lenses** (not one
literal scalar, not merely softening in place).

## Approved decisions

1. **One state, two lenses.** A single source of truth for the two normalized
   coordinates; both lenses (and the calibration metric) read it.
2. **Soften `settled` only; keep `imp` hard-floored.** `imp = max(ℓ − 1, 0)`
   stays — it is what drove `frozen_freeloader_frac` to 0.000 (below-average
   weight ⇒ never freezes). Softening only `settled` is the targeted fix for the
   tail and cannot reintroduce frozen freeloaders.
3. **Pluggable cliff, lead with sigmoid.** `settled(d)` is selectable via a
   `settled_mode` knob; the new variant defaults to the **sigmoid** (the smooth
   version of today's cliff — still pivots at average demand), with steepness `k`
   and the other shapes swept as eval variants. Empirics pick the winner.
4. **No rename.** Keep the stored `confidence` field (renaming ripples through
   history keys, metrics, viz, and ~7 test files for no behavior gain). Document
   it as the *rigidity* lens; expose plasticity `P = 1/(1+confidence)` for
   inspection only.
5. **Selectable, not a rip-out.** Existing `tugofwar` and hard-cliff `twod` rules
   stay intact; the new behavior is A/B-compared, not asserted.

## §1 Shared state — `load` / `demand` helpers (`sprout/currency.py`)

Single source of truth for the two coordinates, so both lenses agree by
construction:

```
def network_scales(net):
    wbar = mean |w|  over live wires   (>= eps)
    Mbar = mean grad_mag over live wires   (>= eps)
    return wbar, Mbar

load(syn, wbar)   = |syn.weight| / wbar          # ℓ  — load it carries now
demand(syn, Mbar) = syn.grad_mag / Mbar          # d  — how hard the loss pushes
```

`update_confidence_2d` and `prune_currency` both compute their coordinates via
these helpers (currently each inlines its own `w̄`/`M̄`). No cross-call caching
is required — the cadences differ (confidence every step, prune every
`t_struct`) and the means are O(synapses) cheap; the win is a **single
definition**, not memoization.

## §2 Lens 1 — Value (pruning) — unchanged behavior

```
V = ℓ + λ·d ;  prune if V < prune_u_floor
```

Identical to today's `prune_currency`, only routed through the shared helpers.
High load **or** high demand protects a wire (a newborn with `w≈0` is protected
by its demand term — no `prune_warmup` needed). No change to grow.

## §3 Lens 2 — Plasticity (learning-rate gate) — softened

```
imp(ℓ)      = max(ℓ − 1, 0)                 # KEEP hard floor (freeloaders never freeze)
settled(d)  = SOFT_CLIFF(d)                  # see §4 — never a hard zero
rigidity    = gain · imp(ℓ) · settled(d)     # in [0, gain·imp_max]
c_ij       <- (1 − alpha)·c_ij + alpha·min(rigidity, c_max)   # EMA, as today
plasticity  = 1 / (1 + c_ij)                 # exposed for inspection
eff_lr      = eta_base · plasticity          # unchanged gate (apply_gated_update)
```

The only change from the current `twod` rule is `settled`: the hard
`max(1 − d, 0)` becomes a smooth, strictly-positive curve. A load-bearing wire
(`imp > 0`) that is briefly contested (`d` slightly above average) now keeps a
*small, load-proportional* rigidity instead of dropping to zero — lifting the
tail off the x-axis and smoothing the confidence↔utility relationship.

## §4 The softened cliff — `settled_mode`

`settled(d)`, selected by `settled_mode`, steepness `conf_k`:

| `settled_mode` | formula | notes |
|---|---|---|
| `hard`    | `max(1 − d, 0)`            | today's cliff (regression baseline) |
| `sigmoid` | `σ(conf_k·(1 − d))`        | **default** — smooth, pivots at avg demand (`d=1 → 0.5`) |
| `exp`     | `exp(−conf_k·d)`           | decays from the quietest wire |
| `rational`| `1 / (1 + conf_k·d)`       | fattest tail (risks under-releasing) |

Lead/default: **`sigmoid`, `conf_k` swept** (start `k≈3`). All four implemented
equally so the harness compares them; `hard` doubles as the regression control
(reproduces current `twod` behavior).

## §5 Config knobs (`sprout/train.py`)

Add to `Config`:

- `settled_mode: str = "sigmoid"`  — `"hard" | "sigmoid" | "exp" | "rational"`
- `conf_k: float = 3.0`            — cliff steepness

`confidence_mode="twod"` continues to select `update_confidence_2d`, which now
reads `settled_mode`/`conf_k`. (`gain`/`alpha`/`c_max` unchanged: `conf_gain`,
`conf_alpha`, `c_max`.) `settled_mode="hard"` + `twod` == today's behavior.

## §6 Eval variants (`evals/spec.py`)

- `currency-2dsoft`   — currency + `confidence_mode="twod"`, `settled_mode="sigmoid"`, `conf_k=3`
- (sweep, added as needed) `currency-2dsoft-exp`, `currency-2dsoft-rational`, `conf_k ∈ {1.5, 3, 6}`

Baseline for the A/B: **`currency-2dconf`** (the current hard-cliff 2D rule),
not legacy.

## §7 Validation

Run via the `running-sprout-evals` skill (`--publish --run-name`):

- **Single-task + shift** (existing regime): watch
  `conf_utility_corr` (target: clear of +0.020, ideally strongly positive),
  `oscillation_frac` (the standing regression — softer settled keeps more wires
  partially rigid, so this may *improve*), `frozen_freeloader_frac` (must stay
  ≈0 — guards the `imp`-floor decision), `final/recovered_test_acc` and
  `auc_test_acc` (must stay ≈ — no accuracy cost).
- **No-shift run** added so `conf_utility_corr` is measured without the
  post-shift EMA-lag artifact (confidence is a slow EMA; the shifted-set demand
  is instantaneous — measuring at the end of a shift run mixes the two).

## §8 Test plan (TDD)

`tests/test_currency.py`:
- `network_scales` / `load` / `demand` return the expected normalized values; `>= eps` guarded on an empty/zero-weight net.
- `prune_currency` and `update_confidence_2d` produce identical numbers to the pre-refactor versions when `settled_mode="hard"` (pure refactor — no behavior change on the control).
- each `settled_mode` matches its formula at sample `d` (e.g. `sigmoid(d=1)=0.5`, `exp(d=0)=1`, all strictly `>0` for finite `d`, monotone decreasing in `d`).
- a high-load, above-average-demand wire (`ℓ=2, d=1.5`) gets **nonzero** rigidity under `sigmoid`/`exp`/`rational` but **zero** under `hard` (the tail-lift, pinned).
- a below-average-weight wire (`ℓ<1`) gets `imp=0` ⇒ rigidity 0 under every mode (freeloader guard preserved).

`tests/test_eval_spec.py`:
- `currency-2dsoft` variant exists with `confidence_mode="twod"`, `settled_mode="sigmoid"`, `conf_k=3`.

`tests/test_train.py` (or config test): `Config` defaults `settled_mode="sigmoid"`, `conf_k=3.0`.

## Non-goals

- No literal single-scalar collapse (shown impossible for both consumers).
- No rename of `confidence`.
- No change to grow (Readout C) or to the prune formula.
- No durability/sticky-freeze mechanism (this is calibration, not
  forgetting-resistance — same boundary as the 2D confidence redesign).

## Risks / open questions

- The tail-lift is **modest** by construction (it scales the already-small
  rigidity of contested load-bearers); `conf_utility_corr` may rise only
  slightly. The no-shift run is the cleaner test of whether the *shape* of the
  relationship improved, separate from the shift artifact.
- If a softer cliff keeps too many wires partially rigid, `oscillation_frac` and
  accuracy are the canaries; `conf_k` (sharper) and `settled_mode` are the dials.
