# Startle: demand-triggered growth (the third phase)

**Date:** 2026-06-11
**Status:** experiment (opt-in `Config.startle`; variants `phasic-startle`,
`phasic-startle-recycle`; promotion decided by eval)
**Builds on:** phasic structural plasticity (2026-06-06) + the recycling
negative result (2026-06-11 sleep-recycling spec)

## Problem: plateau-gated bursts are demand-blind

The recycling experiment isolated the root cause of phasic's continual-regime
losses: **structural change only fires at settledness plateaus, but hiring
signals only exist during demand spikes.** By the time the loss re-settles
after a task transition, the transition's large deltas — the very signal the
grow scan prices ghosts with — have been learned away. Measured consequences:

* zero rehires of recycled blanks in 51/51 attempts (recycle-continual);
* phasic grows ~half as much as continuous (n_grow ~57 vs ~175 on continual)
  and loses second-task acquisition (`b_learned` 0.973 vs currency 0.983 ▲);
* a hidden detector artifact: after a transition the loss-EMA never beats the
  *old* task's `best`, so `since_improve` keeps counting through the descent
  and a burst falsely fires ~`sleep_patience` steps in — a mistimed prune
  while actively learning.

## Design: one loss state machine, two thresholds, three phases

The settledness detector already maintains the only clean signal (the loss
EMA). Startle is its mirror — same state, opposite threshold:

* **settled** (existing): no `tol`-improvement for `patience` steps
  → sleep burst: **prune the weak + grow the wanted + recycle corpses**
  (scheduled maintenance, at `t_struct` ticks).
* **startled** (new): `loss_ema > best × (1 + startle_tol)` sustained for
  `startle_patience` consecutive steps, past warmup
  → startle pass: **grow only**, fired **immediately** (any step — it is an
  alarm, not maintenance).

This gives each structural operation its natural trigger: you know what is
*useless* only when calm (utility meters are clean at a plateau); you know
what is *missing* only when the loss screams (virtual gradients are
informative during a spike). Wake = learn, sleep = consolidate, startle =
hire. Biological reading: arousal-gated plasticity (locus-coeruleus surprise
signals triggering rapid synaptogenesis) alongside sleep-dependent pruning.

### The startle pass

Reuses the burst grow verbatim: score ghosts on a `virt_batch` sample of the
*current* task, grow every candidate above `grow_bar_frac × ref`, uncapped.
The relative bar self-normalizes during a spike — `ref` (mean live |grad|)
rises with the deltas, so only ghosts the loss wants ≫ a typical live wire
clear; no special startle budget is needed. No pruning (spike-time meters are
polluted: load is stale, demand is transient — and the autopsy showed
spike-time pruning kills units being repurposed). No recycling (corpse
detection is timing-insensitive; sleep keeps it).

After firing, `detector.reset()` — the same re-baseline a sleep burst does:
`best ← loss_ema`, counters zeroed. Three effects in one call:

1. **refractory**: a new startle needs a fresh sustained spike *above the
   rebased best* — a stalled-high loss cannot re-fire (a stall is a plateau:
   sleep's job), only further *deterioration* can (escalating crisis);
2. **kills the false-settled artifact**: with `best` rebased to the spike,
   the descent that follows registers as improvement, so sleep now fires at
   the new task's *true* plateau instead of mid-descent;
3. **symmetry**: every structural event (sleep or startle) demands fresh
   evidence before the next.

### Detector changes (sprout/sleep.py, backward-compatible)

`SettlednessDetector(beta, tol, patience, warmup, spike_tol=0.5,
spike_patience=50)` gains `since_spike` (incremented in `update()` while
`loss_ema > best*(1+spike_tol)`, else zeroed) and a `startled` property
(`last step ≥ warmup AND since_spike ≥ spike_patience`). `update()`'s
signature/return are unchanged; `reset()` additionally zeroes `since_spike`.
Variants with `startle=False` never read `startled` — bit-identical.

### Trainer changes (sprout/train.py, phasic-only)

Three new `Config` fields: `startle: bool = False`, `startle_tol: float =
0.5`, `startle_patience: int = 50`. In `_step_currency`, after the detector
update and the (unchanged) `t_struct` rewire block:

```
if cfg.phasic_structure and cfg.startle and cfg.enable_grow \
        and self.sleep_detector.startled:
    n_grown += self._startle_grow()      # any step; logs a "startle" event
```

Placed *after* the rewire block so a sleep burst that just fired (and reset
the detector) cannot double-fire a startle in the same step. RNG is drawn
only when a startle actually fires, so flag-off runs are bit-identical.

### Timing arithmetic (why the defaults)

`beta=0.01` (~100-step memory): a task transition lifts the raw loss to
~0.7–3.5 from a settled ~0.05–0.15, so the EMA crosses `1.5×best` within
~3–10 steps; +50 patience ⇒ the startle fires **~60 steps after onset**,
while `b_steps_to_80 ≈ 200` — squarely inside the hot window. The plateau
EMA's noise band (sample std / √100) sits well inside ×1.5, and a noise
excursion sustaining 50 consecutive smoothed steps is vanishingly unlikely —
single-task runs should fire ~0 startles (verified by smoke test before the
suites; `startle_tol` raised if not).

**Known risk — post-burst spike:** an uncapped sleep prune can itself bump
the loss; if the bump exceeds the rebased `best × 1.5` for 50 steps, a
startle fires right after sleep, regrowing what was cut (prune→grow churn).
Defenses: the prune floor (1.0) only removes near-useless wires (small
bumps), and the 3× grow bar — the proven anti-thrash lever — filters the
regrow bids. `max_regrow` / `oscillation_frac` police this in the eval;
the smoke test checks single-task startle counts first.

## Evaluation plan

Variants: `phasic-startle` (= phasic + startle) and `phasic-startle-recycle`
(+ `recycle_dead`: blanks born at sleep, bidding into hot startle windows —
the full triad). Baseline `phasic`; `currency` as the continuous reference
(its `b_learned` 0.983 ▲ is the number to match). 5 seeds, `--no-cache`
(schema gains `n_startle_events`).

1. **Single + shift** (`startle-vs-phasic`): do-no-harm pre-shift (zero
   startles expected before the swap ⇒ identical trajectories); the label
   swap is a spike, so the post-shift window is a live test — hypotheses:
   `recovered_test_acc` / `recovery_gap` / `final_acc_stability` improve
   (phasic's other honest cost).
2. **Continual A→B→A+B** (`startle-continual`), the primary. Hypotheses:
   H1 startles are orderly (~1–3 per transition, ~0 elsewhere);
   H2 `b_learned` closes on currency (≥0.98) with `b_steps_to_*` ▲;
   H3 the A+B phase startle (forgetting = a spike on old-task samples)
   improves `consolidation`/`relearn_gap`;
   H4 with recycling: `recycled_rehired_frac` finally > 0 and
   `idle_unit_frac` < phasic-recycle's 0.42 (blanks bid hot); honest doubt:
   a blank's bid is `|delta_post|·r_target` vs `3×mean(|delta·a|)` — even
   hot, it needs `delta_post ≳ 5–6× mean`, so zero rehires remains possible
   while live-unit ghosts win the same auctions (still a capacity win);
   H5 costs to watch: synapse count up (hiring), churn (`oscillation_frac`,
   `max_regrow`, `turnover`, `freeloader_frac`) from startle-grown wires
   later pruned at sleep.

Promotion bar: H2 (the headline) with no clear churn/sparsity regression on
(2) and no degradation on (1).

## Alternatives considered

* **Grow at plateaus with a lower bar** — wrong signal: plateau deltas are
  the leftovers the net cannot fix, not the missing-capacity signal.
* **Continuous grow + phasic prune** — abandons the phasic win (grow churn
  returns); startle is the bounded version (grow exactly when demand exists).
* **Two separate detectors (fast/slow EMA change-point)** — more state, same
  behavior; the shared `best` is what makes the refractory and false-settled
  fix fall out for free.
* **Startle prunes too** — rejected: spike-time utility is polluted, and the
  autopsy showed transition-time pruning is exactly what kills repurposable
  units.
