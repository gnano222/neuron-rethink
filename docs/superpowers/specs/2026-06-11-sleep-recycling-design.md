# Sleep-time recycling: apoptosis completes, then neurogenesis

**Date:** 2026-06-11
**Status:** experiment (opt-in variant `phasic-recycle`; promotion decided by eval)
**Builds on:** phasic structural plasticity (2026-06-06 consolidation spec)

## Problem: dead units are an absorbing state

The phasic-vs-continuous eval showed phasic roughly doubles permanently-dead
hidden units (`dead_unit_frac` 0.09 → 0.18 ▼; the old sleep overlay was worse
still at 0.23). A per-seed autopsy (3 seeds, probe every 100 steps,
cross-referenced with the event log) found:

* **~65% of deaths are burst-kills**: the unit was alive immediately before a
  sleep burst pruned its incoming wires and dead immediately after. ~25% died
  during wake from SGD drift (classic dying ReLU, incl. post-shift), ~10% were
  stillborn (never fired since init; bias frozen at exactly 0).
* **The end-state is universal**: fan-in 1 (the orphan guard's floor) where the
  surviving wire *cannot excite* the unit — usually a high-|w| inhibitory-only
  wire, occasionally an excitatory wire from a pre that is itself dead (cascade).
  The orphan guard preserves topology, not function, and strands one zombie wire
  per corpse (~9% of end-state edges).
* **Death is permanent by construction** — the ratchet has four teeth:
  `relu' = 0` ⇒ `delta = 0` ⇒ (1) bias and remaining weights get zero gradient
  forever; (2) never in `active_post`, so ghosts *into* it are never scored
  (the deliberate "no growth wasted on corpses" rule); (3) `a = 0`, so never in
  `active_pre` and ghosts *from* it are never scored; (4) within a burst,
  `_rewire_phasic` prunes → scores → grows, so a unit silenced by this burst's
  prune is already invisible to the same burst's grow.

A dead unit is not a *worthless* firm, it is an *unpriceable* one: `delta = 0`
is zero information, not zero value. Single-task accuracy is preserved (corpses
don't correlate with accuracy across seeds), so the cost is **capacity**: the
continual regime needs to recruit fresh units for task B, and 18–25% of the
width cannot bid.

## Design: make corpses visible again; let the market do the rest

One new operation, at sleep only, opt-in via `Config.recycle_dead` (default
False — the promoted phasic default is unchanged until the eval says otherwise).

At each phasic burst, after the prune and before the grow scan:

1. **Detect** corpses with the existing (previously vestigial) firing-rate EMA:
   a hidden unit with `firing_rate < 1e-6` has not fired for hundreds of wake
   steps (r decays ×(1−β) per silent step; a settledness plateau guarantees
   ≥`sleep_patience` wake steps since the last burst, so truly dead units have
   long collapsed below any rare-but-alive unit). Zero extra forward passes.
2. **Let death complete**: remove ALL the corpse's remaining wires (in + out).
   This reclaims the orphan-guard zombie wire and guarantees the rebirth cannot
   perturb the function (no outgoing wires ⇒ no influence). Grace does not
   apply: this is not the prune lens valuing a live wire, it is cleanup of
   provably-dead circuitry.
3. **Rebirth as a blank**: `bias ← r_target` (reusing the existing constant —
   no new numeric knob), `firing_rate ← r_target` (seed the meter at its new
   steady state). The unit now fires a faint constant and is **visible**:
   it enters `active_pre`, so the very same burst's grow scan prices ghost
   wires *from* it at `|delta_post| · r_target` — and it must clear the same
   selective `grow_bar_frac` as any candidate. If hired, real gradient flows
   back, `delta ≠ 0`, it enters `active_post`, acquires inputs, and SGD
   differentiates it into a feature. If never hired, it sits costless (no MACs)
   and keeps bidding at every future burst.

The "no growth wasted on corpses" invariant survives literally: growth still
never targets a zero-delta unit — we resurrect first, then let the economy
decide. Burst-kills are detected one burst late (their EMA only decays during
the following wake), which doubles as a one-cycle refractory.

### Why these choices

* **Firing-rate EMA over a batch probe**: free (the meter is maintained every
  step and currently unused), and temporally *right* — "has not fired in
  hundreds of steps at a plateau" is what "dead" means here. A 32-sample probe
  risks false-positive recycling of rare-firing live units.
* **bias = r_target, not a new knob**: bid strength scales with the rebirth
  bias (`score = |delta_post| · bias`); r_target (0.15) is on the scale of real
  hidden activations (~0.22 mean), so blanks bid like a typical unit and the
  3× bar still demands genuine starvation before hiring. Only flag added:
  `recycle_dead`.
* **Recycle before the grow scan**: wake-deaths and stillborns get to bid in
  the same burst they are recycled.
* **Phasic-only**: the continuous path is a pinned A/B baseline; recycling is
  implemented inside `_rewire_phasic` only.

### Honest measurement (the metric recycling could game)

A recycled-but-unhired blank *fires*, so `dead_unit_frac` drops ~trivially.
New honest metrics:

* `idle_unit_frac` ↓ — hidden units that are dead **or** have `fan_out == 0`
  (not participating in the function). This is the headline capacity metric;
  recycling only wins if idle capacity falls (or stays flat while continual
  metrics improve).
* `n_recycle_events` (neutral) — mechanism activity.
* `recycled_rehired_frac` (neutral) — of distinct units ever recycled, the
  fraction ending the run non-idle (hired and still in service). NaN for
  variants without recycling.

## Evaluation plan

Variant `phasic-recycle` = the `phasic` config + `recycle_dead=True`. Baseline:
`phasic` (the thing being fixed); `currency` included as the continuous
reference. 5 seeds, w16, `--no-cache` (metric schema changed).

1. **Single + shift** (`recycle-vs-phasic`): 15k + 3k label swap. Guard rails:
   accuracy/sparsity ≈ phasic; hypothesis: idle_unit_frac ▲(lower), post-shift
   stability no worse (blanks as shock absorbers).
2. **Continual A→B→A+B** (`recycle-continual`): the primary. This also closes
   outstanding #1 (dead-unit cost on the forgetting regime, phasic vs
   currency). Hypothesis: recycled capacity improves task-B acquisition
   (`b_learned`, `b_steps_to_90`) and/or consolidation, at ≈ task-A metrics.

Promotion criteria: no accuracy/sparsity regression on (1), a clear win on
idle capacity or any continual headline on (2), no oscillation regression.

## Alternatives considered

* **Homeostatic bias nudging** (prevent death): gentler but fights SGD on bias,
  adds a rate knob, leaves faint zombies costing forward compute.
* **Leaky deltas in the appraisal only**: prices corpses counterfactually, but
  a weight-0 rescue wire into a still-zero-delta unit never moves — needs
  non-zero birth weights (disruption) or full leaky ReLU (changes the function
  class and the pinned validate.py baseline).
* **Neuron-level economics (Readout D)**: liquidate + reallocate width where
  demand concentrates. The most SPROUT-shaped endgame, but a new mechanism
  class; revisit if recycling shows blanks are systematically hired at demand
  hot-spots.
* **Rebirth with random dendrites** (1–2 He-init input wires, bias 0): stronger,
  more diverse bids; kept as a follow-up sweep if bias-only blanks are never
  rehired.
