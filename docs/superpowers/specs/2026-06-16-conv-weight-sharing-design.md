# Convolution / weight-sharing for SPROUT — design

**Date:** 2026-06-16
**Branch:** sprout-v1
**Status:** Phase 1 spec (measure); Phase 2 sketched, gated on Phase 1.

## Motivation

The constant-compute sweeps established that SPROUT's ~0.93 test-accuracy ceiling
on 14×14 MNIST is **architectural**: more width plateaus, depth hurts, more edge
budget self-prunes away, higher resolution hurts at fixed budget — only more data
helps. The one untested lever is **translation-invariant pattern detection**
(convolution / weight-sharing): learn a feature detector once and reuse it at
every image location.

This fundamentally cuts against SPROUT's core assumption — **every synapse is an
independent economic agent** that owns one `weight` and one gradient meter
(`grad_mag`/`grad_signed`), with confidence/prune/grow all reading that wire's
*personal* load and demand. Weight-sharing is the opposite: **many** synapses
share **one** weight (a filter applied everywhere), and "identify a pattern
anywhere" *is* that sharing.

## Decision: measure, then build

Chosen path (user): **measure first, build only if it pays off.**

- **Phase 1 (measure):** put *fixed* (untrained) translation-invariant features in
  front of SPROUT and ask: does test accuracy break ~0.93, and by how much? A
  go/no-go gate, not a contribution.
- **Phase 2 (build, gated):** the faithful "net grows its own filters" version,
  where weight-sharing lives inside the currency economy.

The key enabling insight: **if the front-end filters are fixed, the conv layer is
pure preprocessing** — it produces feature maps, we flatten them, and SPROUT
trains on those instead of raw pixels. So Phase 1 is *one new dataset variant*,
with **zero changes** to the network, the currency economy, the runner's training
loop, or the array backend. The moment filters need to *learn*, that breaks
(they'd need gradients), which is exactly why Phase 1 stays fixed-filter.

## Phase 1 — architecture

```
14×14 image → conv(bank of fixed 3×3 filters, valid) → ReLU → 2×2 max-pool → flatten → standardize → SPROUT
   (196)          6 filters → 6×(12×12)                              6×(6×6)=216          216         (unchanged economy)
```

- **Hand bank (6 filters):** 4 oriented edges (0/45/90/135°, Sobel-like) + 2
  center-surround (blob on/off) — the classic V1 detectors. Each ℓ2-normalized.
- **Random bank (6 filters):** zero-mean, unit-norm 3×3 Gaussian kernels, fixed
  seed. Same count, for a free hand-vs-random A/B.
- **Max-pool buys the translation tolerance:** a feature detected anywhere in a
  2×2 window collapses to one value, so "is there an edge around here" stops
  caring about the exact pixel. Pool size is the invariance knob (bigger = more
  position-invariance, coarser map).
- **Standardize the conv features on TRAIN stats** (reuse `_standardize_on_train`;
  the eps guard handles near-constant feature columns).

The bank is part of the *fixed architecture*, not the data draw: it is built with
a fixed seed and applied identically across all data seeds, so the feature space
is a deterministic transform.

## Phase 1 — code touch-points

| File | Change |
|---|---|
| `sprout/conv_features.py` *(new)* | `filter_bank(kind, seed, size)` → list of kernels; `conv_features(images, bank, pool, nonlin)` → flattened feature matrix. Pure NumPy (vectorized via `sliding_window_view`), no training. |
| `sprout/datasets.py` | `mnist_conv_transform(X_flat, side, bank_kind, pool, nonlin)` (pure, testable); `load_mnist_conv_split(...)`; register `mnist-conv`, `mnist-conv-rand`, `mnist-full-conv` in `get_dataset`. |
| `sprout/train.py` | Add `Config.init_dataset: str | None = None` (per-variant dataset override; sibling to `init_layers`/`init_density`; Trainer ignores it, only the runner consults it). |
| `evals/runner.py` | `_dataset_name(cfg, spec)` helper (override → else `spec.dataset`); `run_one` uses it. (`init_dataset` is already inside `asdict(cfg)` in `_cache_key`, so caching stays correct.) |
| `evals/cli.py` | Add the conv dataset names to `--dataset` choices. |
| `evals/spec.py` | `_sparse`/`_dense` accept an optional `dataset=`; add variants `mnist-conv-hand`, `mnist-conv-rand` (both `_sparse((216,32,10),0.5)` on the conv datasets); add their names to `_BOUNDED_GROW_VARIANTS`. |

**Untouched:** `network.py`, `currency.py`, the Trainer loop, and `fast.py`
(array backend). That is the whole point of choosing fixed filters.

## Phase 1 — measurement protocol & decision gate

Run via the existing `--publish` harness (running-sprout-evals protocol), **5
seeds**, primary on **14×14 MNIST**:

- One suite, arms: `mnist-w32-sparse` on **raw** `mnist` (baseline) vs
  `mnist-conv-hand` vs `mnist-conv-rand`. Same self-rewiring architecture
  (phasic-startle-k4, w32-sparse); the *only* difference is the input features.
- Report **final/max test acc** (does it clear ~0.93 with a bootstrap-CI ▲?) and
  **edge-steps**. The conv arm has ~10% more edges (216 vs 196 input), and the
  conv itself is a one-time precompute (per-step SPROUT compute unchanged) — both
  stated explicitly, not hidden.
- Secondary: **hand vs random** — does designed structure matter, or do *any*
  translation-invariant features suffice?
- **784 follow-up** only if 14×14 shows a win (conv's payoff is larger there).

**Gate:**
- Clear ▲ that breaks ~0.93 → **green-light Phase 2.**
- `≈` or marginal → the ceiling is likely data/other; reconsider before building
  the hard version. (This is the entire reason to measure first.)

## Phase 2 — sketch (gated on Phase 1)

If Phase 1 pays off, make weight-sharing a first-class citizen of the economy:

1. **Geometry:** input neurons carry real pixel `(row,col)` coordinates so the net
   knows locality.
2. **`Filter` (template) object:** a shared small weight vector (e.g. 9 taps) with
   **one** currency meter aggregating gradient over all its placements.
3. **`Synapse` becomes a binding:** references `(template_id, tap, pre, post)`;
   effective weight = `template.weights[tap]`. Backward sums each template's
   gradient across all placements; SGD updates the template once.
4. **Two-level economy:** grow/prune **filters** ("does this detector earn its
   keep?") *and* **placements** ("apply filter F at location L?" — the RigL
   virtual-gradient growth rule, but for a placement).
5. **Array backend:** gather/scatter with shared weights (the bulk of the work).

Specced in full only after the Phase 1 gate is green.

## Risks / notes

- **Compute honesty:** fixed-filter conv is free at train time (precomputed), but
  the conv arm's net is ~10% larger (216 vs 196 input). Report edge-steps; if the
  win is marginal, re-run the conv arm at matched edges (lower `init_density`).
- **Pooling strength:** 2×2 is mild. Digits are roughly centered, so strong
  invariance may be unnecessary; pool size is a cheap follow-up knob if needed.
- **Polarity:** ReLU after a signed filter keeps one polarity; the hand bank
  includes both blob-on/off, and `nonlin="abs"` is available as a knob.
```
