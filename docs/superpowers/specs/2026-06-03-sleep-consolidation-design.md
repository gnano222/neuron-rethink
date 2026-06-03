# Settledness-gated sleep consolidation — design

**Date:** 2026-06-03
**Branch:** sprout-v1
**Status:** approved, ready for implementation

## Goal

Cut the network's active-synapse count ~losslessly by pruning aggressively
**only when the network has settled** — a "sleep" consolidation burst — instead
of pruning hard continuously (which churns the net and destroys accuracy).

## Motivation (what the measurements showed)

A chain of measurements on the default w16 net (`(2,16,16,16,2)`, currency,
spirals) reframed the problem and pointed at this design:

1. **Pruning ⊥ activation fraction.** Cranking the prune rate cut edges 240→54
   but left the firing fraction flat (~46%→44%) and destroyed accuracy
   (0.997→0.56). Pruning is a *connection*-sparsity lever, not an activation one.
2. **The redundancy is low-rank, not pairwise.** Hidden layers have effective
   rank ~2–3 (16-wide), but synapse-level redundancy-merge is lossless only for
   ~14% near-duplicate wires and never beats magnitude pruning — the low-rank
   structure isn't reachable by pairwise folding without a basis change that
   would dissolve the explicit-neuron design.
3. **The real lever is *timing*, not the criterion.** One-shot magnitude pruning
   of the *converged* net removes ~27% of synapses for free (acc 0.993) and is
   far gentler than the same sparsity reached by continuous online churn
   (`prune-brutal` ended at 77% removed → 0.56). "Aggressive pruning hurts" was a
   churn artifact, not the wires being irreplaceable.
4. **The settled state is real and long.** Loss settles below ~0.05 by ~step
   4000 of 15000 and stays there — a long, safe consolidation window.
5. **Loss is the only clean settledness signal.** Mean confidence is
   non-monotonic and noisy; `M̄` (mean grad magnitude) spikes 8–280%/window;
   smoothed **loss** is the usable detector input.

## Architecture

A new opt-in mechanism, **off by default** (baseline unchanged). Two units:

### Unit 1 — `SettlednessDetector` (`sprout/sleep.py`)

An early-stopping-style "patience" detector on an EMA of the per-step training
loss. One responsibility: answer *"has the network settled?"* each step.

```python
class SettlednessDetector:
    def __init__(self, beta, tol, patience, warmup):
        # beta: EMA smoothing of loss (~1/beta step memory)
        # tol: relative improvement that counts as "still improving"
        # patience: steps without a tol-improvement before declaring settled
        # warmup: no "settled" verdict before this step
        ...

    def update(self, loss: float, step: int) -> bool:
        """Feed one step's loss; return True iff settled *now*."""
        # loss_ema <- (1-beta)*loss_ema + beta*loss   (seed with first loss)
        # if loss_ema < best*(1-tol): best = loss_ema; since_improve = 0
        # else:                       since_improve += 1
        # return step >= warmup and since_improve >= patience

    def reset(self):
        """After a consolidation burst: require a fresh plateau before the next.
        Sets since_improve = 0 and best = current loss_ema (or inf if unseen)."""
```

`reset()` is the self-pacing safeguard: each sleep requires the loss to re-settle
afterward, so a burst that perturbs the net cannot trigger another until it
recovers — making churn structurally impossible and handling the label-swap /
continual regimes for free (a shift re-raises loss → detector waits → consolidates
again only after re-convergence).

### Unit 2 — sleep wiring in `Trainer` (`sprout/train.py`)

- `Trainer.__init__`: when `cfg.enable_sleep`, build `self.sleep_detector` and
  init `self.settled = False`.
- Per step (currency path only): after the loss is known, `self.settled =
  self.sleep_detector.update(loss, self.step_idx)` (cheap, every step).
- At the existing rewire cadence (`step_idx % t_struct == 0`), in
  `_rewire_currency`: if `cfg.enable_sleep and self.settled` →
  **consolidation burst**:
  - prune with the aggressive params (`sleep_prune_floor`, `sleep_max_prune`)
    by reusing `prune_currency` (the criterion is fine; only timing/aggression
    change),
  - **skip `grow` this round** (don't explore while consolidating),
  - record a `{"type": "sleep"}` event, and call `self.sleep_detector.reset()`.
  - Otherwise: normal wake (gentle `prune_currency` at `prune_u_floor` + `grow`).
- Reversibility: `grow` stays on during wake, so an essential wire over-pruned by
  a burst can regrow.

## Config (new fields, all opt-in)

```python
enable_sleep: bool = False        # master switch; baseline unchanged
sleep_warmup: int = 2500          # no consolidation before this step
sleep_loss_beta: float = 0.01     # loss EMA smoothing (~100-step memory)
sleep_loss_tol: float = 0.03      # rel. loss improvement that resets the plateau
sleep_patience: int = 1500        # steps w/o a tol-improvement => settled
sleep_prune_floor: float = 2.0    # aggressive prune utility floor during a burst
sleep_max_prune: int = 10         # aggressive per-burst prune cap
```

Defaults chosen from the measured trace: first burst lands ~step 4000 (settled
region), bursts re-pace via `reset()` + re-plateau.

## Eval variant (`evals/spec.py`)

Add `"sleep"`: the `currency` baseline config + `enable_sleep=True` (+ the sleep
defaults above). Compared against the `currency` baseline.

## Validation

Multi-seed `evaluate.py` run, `sleep` vs `currency` baseline, `--publish`:

- **Win:** lower `synapse_count_end` / `effective_density` and lower
  `ghost_pairs_scored` (cheaper grow scan on a smaller net) at **equal**
  `final_test_acc` / `auc_test_acc`.
- **Watch (honest losses):** `dead_unit_frac` (over-consolidation), and — in the
  shift / continual regimes — `recovered_test_acc` and `forgetting` (did
  consolidating mid-run cost adaptability?).

Also re-run `validate.py` (must stay 7/7 — it uses the pinned default config with
`enable_sleep` off, so it must be unaffected) and the full `pytest` suite.

## Risks & mitigations

- **Detector misfire (noisy loss).** Mitigated by EMA smoothing + warmup floor +
  patience (a single dip can't trigger).
- **Over-consolidation → dead units / accuracy loss.** Mitigated by `grow`
  recovery during wake, the bounded per-burst cap, the `reset()`-gated re-pacing,
  and the eval metrics above (which surface it honestly).
- **Determinism.** The detector reads only the per-step loss and step index; no
  new RNG draws, so cached/seeded runs stay reproducible.

## Out of scope (YAGNI)

- New pruning *criteria* (magnitude/utility already suffices — finding #3).
- Neuron-level merge / PCA consolidation (low-rank redundancy unreachable without
  breaking the explicit-neuron design — finding #2).
- Activation-fraction mechanisms (k-WTA, metabolic loss) — a separate axis,
  parked.
