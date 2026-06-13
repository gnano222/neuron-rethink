# SPROUT 2D (importance × settled) confidence — design

**Date:** 2026-05-31
**Branch:** sprout-v1
**Status:** approved, ready for implementation

## Goal

Re-derive `confidence` so it is **calibrated to wire utility**. Today's currency
confidence (`update_confidence_currency`) earns from *low gradient + consistent
direction* and **ignores weight magnitude entirely**, while the prune-utility it
is implicitly meant to track is `U = |w|/wbar + lam·M/Mbar`. The result is a
structural anti-correlation (`conf_utility_corr ≈ −0.17/−0.19`): confidence
freezes wires by settledness regardless of whether they carry any load, so it
freezes *freeloaders*.

Fix: make confidence read the **same 2D state prune reads** — *importance*
(weight) and *settledness* (demand) — so a wire is frozen only when it is both
**important AND settled**.

## Approved decisions

1. **Contention behavior: releases.** When a settled wire is later contested by
   a new task (demand spikes, `d > 1`), confidence falls and the wire unfreezes.
   This is a **calibration** redesign, not a durability/stickiness one. (Sticky
   freeze for forgetting-resistance is explicitly out of scope.)
2. **Formula: B2 — only above-average-weight wires freeze.** The importance term
   gates on weight *above the mean*, so the settled bulk stays plastic and only
   genuinely load-bearing wires lock up (less over-freezing).
3. **Selectable, not a rip-out.** Add the new rule behind a `confidence_mode`
   switch; the existing tug-of-war stays intact. Add an eval variant so the two
   are A/B-compared on `conf_utility_corr` rather than asserted.

## §1 Mechanism — `update_confidence_2d(net, gain, alpha, c_max, eps)`

New pure function in `sprout/currency.py`, run every step on every live wire:

```
wbar    = mean |w| over live wires          # same scale prune_currency uses
Mbar    = mean grad_mag over live wires
for each wire:
    imp     = max(|w|/wbar - 1, 0)           # importance: ABOVE-average weight
    settled = max(1 - M/Mbar, 0)             # settledness: below-average demand
    target  = min(gain * imp * settled, c_max)
    c      <- (1 - alpha)*c + alpha*target    # EMA toward target
```

`apply_gated_update`'s `eta_eff = eta_base/(1+c)` is **unchanged** — confidence
still only throttles the per-wire learning rate.

**Why it is correct on the edge cases (no `m_floor`/`kappa` needed):**
- below-average-weight wire → `imp = 0` → never freezes (freeloaders ignored).
- dead but heavy wire (M≈0, |w| large) → settled & important → frozen (correct:
  it carries load and the loss has stopped pushing it).
- contested wire (M ≫ Mbar) → `settled = 0` → target 0 → confidence decays.

**Calibration payoff:** on settled survivors (where converged wires live),
`confidence ≈ gain·(w_norm − 1)` and `utility ≈ w_norm`, monotonically related,
so `conf_utility_corr` should swing strongly positive.

## §2 Config knobs (`sprout/train.py`)

- `confidence_mode: str = "tugofwar"` — `"tugofwar"` (existing) or `"twod"` (new)
- `conf_gain: float = 2.0` — maps above-average weight to confidence
  (3× mean weight, fully settled → `c ≈ 4`; 2× mean → `c ≈ 2`)
- `conf_alpha: float = 0.01` — EMA rate (~100-step time constant, matching `beta_g`)
- `c_max` (existing, 5.0) reused as the ceiling.

`Trainer._step_currency` branches: when `enable_confidence` and
`confidence_mode == "twod"`, call `update_confidence_2d(net, cfg.conf_gain,
cfg.conf_alpha, cfg.c_max)`; otherwise the existing `update_confidence_currency`.
No other call sites change.

## §3 Eval variant (`evals/spec.py`)

Add `currency-2dconf`: the `currency` config plus `confidence_mode="twod"`. Lets
the harness compare it against `currency` on `conf_utility_corr` and
`frozen_freeloader_frac` (and confirm accuracy is not hurt).

## §4 Testing (TDD)

Unit tests in `tests/test_currency.py` on `update_confidence_2d`, hand-built nets:
- heavy + settled wire → confidence rises toward a positive target
- average / below-average weight wire → confidence stays ~0
- contested wire (`M ≫ Mbar`) → target 0, confidence decays toward 0
- dead-but-light → never frozen; dead-but-heavy → frozen
- EMA moves exactly fraction `alpha` toward target; result clipped to `c_max`

Spec test in `tests/test_eval_spec.py`: `currency-2dconf` sets `confidence_mode`.
Regression: full suite stays green; the tug-of-war path is untouched.

## §5 Scope

**In:** `update_confidence_2d`, the three Config knobs, the mode branch, the eval
variant, tests. **Out:** stickiness/durability (releases chosen), any prune/grow
change (prune already reads `w_norm + lam·d`), the forgetting benchmark (done).

## Success criteria

1. Full suite green; tug-of-war path unchanged.
2. A published A/B run (`currency` vs `currency-2dconf`): `conf_utility_corr`
   materially higher (target: clearly positive vs ≈ −0.19) and
   `frozen_freeloader_frac` lower, with `final_test_acc` not regressed. Honest
   reporting either way.
