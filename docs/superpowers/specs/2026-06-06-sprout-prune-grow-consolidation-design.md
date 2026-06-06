# Prune/Grow Consolidation: declutter (A) + phasic structural plasticity (C)

**Date:** 2026-06-06
**Branch:** sprout-v1
**Goal:** Collapse the convoluted prune/grow surface into one cohesive mechanism —
*one* prune, *one* grow, fired by *one* trigger — serving the project's north star
(a simple, sparse, efficient net). Two increments: **A** removes the dead legacy
twin; **C** makes structural change *phasic* (wake = learn; sleep = rewire).

> This doc doubles as the implementation plan (task checklist at the end). It was
> produced via the brainstorming process; the two design forks were chosen by the
> user: **cold start = pure phasic**, **rewire budget = independent thresholds**.

---

## Background: what exists today (the convolution)

Two complete, parallel structural-plasticity stacks selected by `grad_currency`:

| | Legacy (`grad_currency=False`) | Currency (`grad_currency=True`, live default) |
|---|---|---|
| Prune | `plasticity.prune` `|w|·r_pre` | `currency.prune_currency` `load+λ·demand` |
| Grow | `plasticity.grow` underfiring+activation | `currency.grow_currency` RigL virtual-gradient |
| Confidence | `learning.update_confidence` 3-factor | `currency.update_confidence_2d` |
| Homeostasis | `plasticity.homeostasis` | — |

On top of the currency pruner sits a **sleep overlay**: a `SettlednessDetector`
plateau-gate that swaps the prune floor `0.5→1.0`, lifts the cap `2→all`, and
**skips grow** when settled. So the live system has **two structural modes**
(continuous churn every `t_struct` + aggressive plateau bursts) and ~30 knobs.

**Diagnosis:** prune and grow answer the same question ("should this wire exist?")
from opposite sides, but are split across two functions/thresholds/caps fired on
the same tick — and that split is the source of the grow↔prune oscillation, which
is then patched three separate ways (inflated `grow_bar_frac`, `ghost_meter`,
sleep timing). Re-unify the *trigger* and the patches dissolve.

---

## Part A — Declutter (increment 1, behaviour-preserving for the currency path)

Make the currency stack the *only* stack. Delete the legacy twin.

### Delete
- `sprout/plasticity.py` (whole file: `prune`, `grow`, `homeostasis`).
- `sprout/learning.py`: `update_eligibility`, `update_confidence` (3-factor).
  **Keep** `update_firing_rates`, `apply_gated_update` (used by currency path/viz).
- `sprout/train.py`:
  - the non-currency `else` branch in `step()`;
  - methods `_prune`, `_grow`, `_homeostasis`, `_update_eligibility`, `_update_confidence`;
  - flags `enable_eligibility`, `enable_homeostasis`;
  - legacy-only `Config` knobs: `lambda_e`, `gamma_q`, `gamma_h`, `e_half`,
    `theta_prune`, `f_under`, `prune_warmup`, `t_homeo`, `grow_budget`, `rho`.
- `validate.py`: `--legacy` mode, `_legacy_eligibility_selectivity`, the legacy
  branch of `_check_growth` (the `from sprout.plasticity import grow` import).
- `run.py`: presets `legacy-step2..6`, `legacy-full`; the `enable_eligibility` /
  `enable_homeostasis` keys in `summarise()`; the `.eligibility` read; the
  `theta_prune` / `prune_warmup` legacy CLI args.
- `evals/spec.py`: the `legacy-full` variant (the only one using `enable_eligibility`).
- `tests/test_plasticity.py` (whole file); the eligibility / 3-factor-confidence /
  legacy-path tests in `tests/test_learning.py`, `tests/test_train.py`,
  `tests/test_infra.py`, `tests/test_eval_spec.py`.

### Keep (deliberately, to bound scope / risk)
- **`grad_currency` field**: retained as a vestigial, default-**True** selector
  (the legacy branch it gated is gone, so it is now inert). Removing it cleanly
  means editing 46 eval-variant lines + tests right before the publish run; that
  churn is deferred to a trivial fast-follow. Documented as deprecated.
- **`Synapse.eligibility` field + viz "glow"**: left in place (always 0 now) to
  avoid cascading into `viz.py`. Flagged as future cleanup.
- **Tug-of-war confidence** (`confidence_mode="tugofwar"`, `update_confidence_currency`,
  `gamma_dec/up/dn`, `m_floor_frac`): kept as an opt-in alt confidence rule
  (orthogonal to prune/grow; low risk; not the user's target).

### Acceptance
Full suite green; **`validate.py` 7/7 unchanged** (currency criteria untouched).
No behaviour change to any currency run.

---

## Part C — Phasic structural plasticity (increment 2, the architecture)

**One signal** (gradient demand, already metered), **one operation** (a rewire
pass = prune the weak + grow the wanted), **one trigger** (a settledness plateau).
Wake = pure gated-SGD + meter the gradient + update confidence; **zero** structural
change. Sleep = the rewire pass. This *subsumes* the sleep overlay rather than
layering on it, and makes oscillation structurally impossible (rewires are far
apart and the detector resets after each).

### New config
```python
# Structural plasticity is PHASIC: wires change only at settledness plateaus.
# True (default) = the C architecture. False = the legacy continuous path,
# retained as the pinned A/B baseline (and validate.py guardrail).
phasic_structure: bool = True
```

### Behaviour (`_rewire_currency`, called every `t_struct` as today)
- **Detector** is built whenever `enable_sleep OR phasic_structure` (phasic always
  needs it — it *is* the trigger). `self.settled` is updated every step in `_step_currency`.
- **`phasic_structure=True`:**
  - `if not self.settled: return (0, 0)` — wake: no structural change.
  - On settled (a "sleep" event):
    1. **Prune** `prune_currency(floor=sleep_prune_floor=1.0, cap=no-cap)` — drop
       every below-floor wire past grace (quality-filtered tail).
    2. **Grow** `grow_currency(bar=grow_bar_frac·ref, cap=no-cap)` — add every
       ghost the loss wants above the relative bar. (No longer skipped — the bar
       self-limits how many.)
    3. `sleep_detector.reset(); self.settled = False` — require a fresh plateau.
  - **Cold start (pure phasic):** no rewire until the first settled verdict
    (≥ `sleep_warmup` 2500 + `sleep_patience`). The net learns on its initial
    `build_graph` topology until then. Falls out of the detector warmup + the
    `not settled` guard — no extra code.
- **`phasic_structure=False`:** the existing continuous path, **unchanged** (gentle
  prune+grow every tick; sleep overlay swaps floor/cap + skips grow when settled).

"No-cap" reuses the established pattern: `cap = len(net.synapses)` when the
per-burst cap is `None`. Order is prune-then-grow (grown wires are age-0 →
grace-protected from the same pass regardless).

### Retired by C (no code, just unused under phasic)
The oscillation patches lose their purpose under phasic timing: `ghost_meter` is
subsumed by the plateau gate (rewires are far apart; the plateau *is* the
persistence filter); `grow_bar_frac` no longer needs its anti-oscillation inflation
(left at 3.0 for now — relaxing it is an eval follow-up, "change architecture
first, re-tune params second").

### Pins (promote default, pin references — established pattern)
- **`validate.py`**: pin `phasic_structure=False` (+ `enable_sleep=False` as today)
  → continuous-primitive guardrail, **7/7 unchanged**, no recalibration.
- **`evals/spec.py` `currency`**: pin `phasic_structure=False, enable_sleep=False`
  → stable continuous baseline (meaning unchanged).
- **`evals/spec.py` `sleep`**: pin `phasic_structure=False` → continuous+sleep =
  today's promoted default = the **A/B baseline** for the publish run.
- **New `evals/spec.py` `phasic`**: `phasic_structure=True` (the new architecture).
- Dormant research variants inherit `phasic_structure=True` (fine — finished
  experiments, same as when sleep was promoted).

---

## Data flow

```
WAKE step (phasic, not settled):
  forward → meter gradient (grad_mag/signed) → update_confidence_2d (gates LR)
          → gated SGD → age++ → detector.update(loss)        [no structure]

SLEEP step (phasic, settled, on a t_struct tick):
  ... wake work ... → REWIRE: prune_currency(1.0,nocap) + grow_currency(bar,nocap)
                    → detector.reset()
```

## What does NOT change
- `prune_currency`, `grow_currency`, `batch_edge_scores`, the 2D confidence rule,
  the gradient meters, gated-SGD LR gating — all identical. C is a *timing* policy.
- The continuous path (now behind `phasic_structure=False`) is byte-for-byte the
  current behaviour.

---

## Testing strategy (TDD)
- **A:** delete legacy tests; assert `import sprout.plasticity` fails / is gone;
  assert `Config` has no legacy knobs; suite green; `validate.py` 7/7.
- **C (new tests in `tests/test_train.py` / `tests/test_sleep.py`):**
  1. `phasic_structure=True` + not settled ⇒ **no** prune/grow events during wake.
  2. Forcing `settled=True` on a `t_struct` tick ⇒ a "sleep" event with **both**
     prune and grow attempted in one pass; detector reset afterward.
  3. Default `Config` has `phasic_structure=True`.
  4. `phasic_structure=False` ⇒ existing continuous behaviour preserved (regression).
  5. Cold start: no structural event before `sleep_warmup`.

## Eval plan (publish)
`evaluate.py --variants currency,sleep,phasic --baseline sleep --dataset spirals`
on the default w16 topology/horizon, with shift, ≥5 seeds, `--no-cache --publish
--run-name phasic-vs-continuous`. Headline question: does **phasic** preserve
accuracy (`final/pre_shift/recovered_test_acc`) while being **sparser**
(`synapse_count`) and **less oscillatory** (`oscillation_frac`) than the
continuous+sleep default? Report honest wins AND losses (watch `dead_unit_count`).
Commit+push the run folder; summarise with the mobile key-metrics table.

## Risks / open items
- **Dead-unit cost on the continual/forgetting regime** remains unmeasured (carried
  over from sleep promotion); the publish run includes a shift to probe recovery.
- **Cold-start handicap**: if the initial graph is poor, early learning runs on it
  until the first plateau. Mitigation lever (if eval shows a deficit): a scheduled
  first sleep — deferred unless the data demands it.
- **Vestigial `grad_currency` / `eligibility` field**: deliberate scope bound;
  trivial fast-follow.

---

## Task checklist (execution order)

**A — declutter**
- [ ] A1. Delete `sprout/plasticity.py` + `tests/test_plasticity.py`.
- [ ] A2. Trim `sprout/learning.py` (drop eligibility + 3-factor confidence) + its tests.
- [ ] A3. `train.py`: remove legacy branch/methods/flags/knobs; `grad_currency`
      default→True (inert); confirm currency path runs unconditionally.
- [ ] A4. Fix `validate.py` (drop `--legacy` + legacy helpers); run → **7/7**.
- [ ] A5. Fix `run.py` (legacy presets, summarise keys, CLI args) + `evals/spec.py`
      (drop `legacy-full`).
- [ ] A6. Fix remaining tests (`test_train`, `test_infra`, `test_eval_spec`,
      `test_learning`); **full suite green**.

**C — phasic**
- [ ] C1. Add `phasic_structure` to `Config` (default True) + default-config test (RED→GREEN).
- [ ] C2. Build detector when `enable_sleep OR phasic_structure`.
- [ ] C3. Phasic branch in `_rewire_currency` (wake-skip; settled ⇒ prune+grow nocap;
      reset) — TDD tests 1,2,5.
- [ ] C4. Pin `validate.py`, `currency`, `sleep` to `phasic_structure=False`; add
      `phasic` variant; regression test 4; **suite green + validate 7/7**.

**Publish**
- [ ] P1. Run the eval with `--publish`; commit+push the run folder.
- [ ] P2. Summarise (mobile key-metrics table + verdict + path + honest wins/losses).
