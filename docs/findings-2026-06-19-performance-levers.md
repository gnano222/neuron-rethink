# Performance levers for SPROUT — what moves accuracy, and why MNIST hid it

**Date:** 2026-06-19 · **Branch:** sprout-v1 · **Task:** improve prediction
accuracy while staying faithful to gradient-as-currency (sparse, self-wiring,
phasic, legible).

## TL;DR

The lever that improves prediction performance is **learned weight-shared filters**
(Conv-SPROUT's core). Four other levers did not help. The reason most "failed" is
the same: **MNIST is saturated** for this model family (~0.94–0.95, tiny overfit
gaps), so no accuracy lever has headroom to show a gain. Moving to a harder
substrate (**Fashion-MNIST**) revealed the truth — learned conv beats both a raw
MLP and a *fixed* conv bank by **~+5pt**, while depth, width, augmentation, and an
optimizer change add nothing.

Every experiment was TDD'd with finite-difference gradient checks, run multi-seed,
and published. All work is additive: `validate.py` stays 7/7 and the promoted
baselines are untouched.

## The levers

| Lever | Substrate | Result | Run folder |
|---|---|---|---|
| Currency-native optimizer (`step ∝ S/M`) | spirals | ✗ regressed (every η) | `opt-currency-spirals-etasweep` |
| Translation augmentation | MNIST-14 | ✗ regressed | `mnist-shift-augmentation` |
| Stacked / deep conv (depth-2) | MNIST-28, Fashion-28 | ✗ no help | `deep-conv-mnist28`, `deep-conv-fashion28` |
| Wide conv | MNIST-28, Fashion-28 | ≈ ties | (same) |
| **Learned conv (Conv-SPROUT)** | **Fashion-28** | ✅ **+4.6–5pt** | `deep-conv-fashion28`, `conv-sprout-fashion28` |

### 1. Currency-native optimizer — regressed
Idea: make the gradient meters the optimizer — step along `S/(M+ε)` (the
consistency coefficient `κ·sign(S)`), a self-normalizing, auto-annealing per-wire
adaptive step that reuses the currency with no new state (`Config.optimizer=
"currency"`, gradient-checked in `tests/test_learning.py`). On spirals it lost at
every η (final, AUC, calibration all down). **Why:** normalizing the step flattens
the `|w|` distribution that confidence/prune read as *importance* `(load−1)₊`; with
load ≈ 1 everywhere, confidence never builds, nothing consolidates, the net drifts
off its peak (`conf_utility_corr` 0.32 → ~0). Elegant, but in tension with the
load-based machinery. Kept as pinned `opt-currency-*` references.

### 2. Translation augmentation — regressed
On-the-fly random shifts (`Config.augment_shift_max`, matched-epoch — the static 4×
expansion `mnist-aug` under-trains at fixed steps). On MNIST-14 accuracy fell
monotonically with shift size (0.903 → 0.850 → 0.662). **Why:** MNIST is
*pre-centered* — the test set has no positional variance to be robust to — and
14×14 integer shifts are coarse/destructive (train accuracy itself collapses). The
baseline's overfit gap was only +0.023: the model wasn't overfitting, so a
regularizer had nothing to fix.

### 3 & 4. Deep / wide conv — built, gradient-checked, no help
A faithful stacked conv (`sprout/conv_deep.py`): multi-channel conv math including
the **input-gradient transposed conv** that lets gradient flow between layers,
`ConvLayer` (multi-channel filter-level currency), `DeepConvModel` (whole-stack
gradient-checked), `DeepConvTrainer`. On MNIST-28 *and* Fashion-28, depth-2 did not
beat depth-1 (it was *worse*: its 2×2 pool collapses the feature map to 400 dims and
starves the head, which prefers depth-1's 1352). A capacity control (`depth1-wide`,
16 filters, no depth) only tied — so it is not a depth-vs-capacity confound. Feature
richness to the head matters more than conv depth on these tasks. The machinery is
correct and reusable; it just is not the lever here.

## The pivotal insight: MNIST is saturated

Across the levers, every overfit gap was tiny (+0.011 to +0.023). The models are
not overfitting — they sit at MNIST's intrinsic ceiling (~0.94–0.95 for the
conv+sparse-head family, at *both* 14×14 and 28×28). **The bottleneck is the
benchmark, not the model**, so no accuracy lever can show a gain. The earlier
"failures" were really "no headroom."

## Fashion-MNIST: the lever appears

Added `fashion` (14×14) and `fashion-full` (784) to the dataset registry (OpenML,
drop-in like MNIST). Fashion is much harder (overfit gap +0.035 — real headroom).
Running raw-MLP vs conv vs deep vs wide in **one codepath** (a `depth0` raw-MLP arm
in `deep_conv_experiment.py`), full-resolution 28×28, 5 seeds:

| arm | test acc | filters | note |
|---|---|---|---|
| raw MLP (`depth0`) | 0.799 | — | floor |
| fixed-hand conv | 0.803 | 6 | **≈ raw MLP** |
| learned conv (`depth1`) | 0.848 | 8 (learned) | **+4.9pt vs raw** |
| conv-sprout (self-sizing) | 0.849 | 7 (12→7) | **+4.6pt vs fixed** |
| deep conv (`depth2`) | 0.822 | 8→16 | up vs raw, worse than depth1 |
| wide conv (`depth1-wide`) | 0.850 | 16 | ≈ depth1 |

The sharp finding: **fixed-hand conv (0.803) ≈ raw MLP (0.799)** — generic
edge/blob filters barely help on clothing. The whole conv gain comes from
**learning** the filters. On MNIST this was invisible (fixed edges already suit
digits, and the task is saturated); on Fashion learned conv beats both the fixed
bank and the raw MLP by ~+5pt. The promoted Conv-SPROUT reaches the same 0.849 at
**7 self-sized filters** — accuracy parity with a plain 8-filter learned conv, with
the efficiency of self-sizing. This validates Conv-SPROUT's central thesis (learn
weight-shared filters inside the economy) on a substrate with headroom.

## Conclusions

- **Learned weight-shared filters are the accuracy lever** for this family; convolution with *fixed* generic filters is not (≈ raw MLP on a harder task).
- **Depth and width on the conv front-end do not help** here; pooling that starves the head of features actively hurts.
- **Regularizer/optimizer tweaks (augmentation, S/M optimizer) do not help** a model that is not overfitting on a saturated benchmark.
- **Benchmark choice dominates lever evaluation.** Test accuracy levers on a task with headroom (Fashion-MNIST), not a saturated one (MNIST).

## Reusable artifacts

- `sprout/conv_deep.py` — gradient-checked stacked-conv machinery (multi-channel conv + input-grad, `ConvLayer`, `DeepConvModel`, `DeepConvTrainer`). Additive; available for a task where composition genuinely matters.
- `Config.optimizer="currency"` and `Config.augment_shift_max` — pinned, opt-in, off by default.
- `fashion` / `fashion-full` datasets — a headroom substrate for future lever evaluation.
- Eval runs: `docs/eval-runs/{opt-currency-spirals-etasweep, mnist-shift-augmentation, deep-conv-mnist28, deep-conv-fashion28, conv-sprout-fashion28}`.
