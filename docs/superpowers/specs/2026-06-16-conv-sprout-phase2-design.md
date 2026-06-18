# Conv-SPROUT (Phase 2): weight-sharing inside the currency economy — design

**Date:** 2026-06-16
**Branch:** sprout-v1
**Status:** Phase 2 build spec. Phase 1 gate came back qualified-green
(`docs/eval-runs/conv-front-end-mnist14`, `…-mnist784`): fixed translation-invariant
features reliably beat raw (+3.7pt 14×14, +4.2pt 784, +2.7pt compute-matched), the
win grows with image size, and hand≈random — so the lever is real, and the open
question is whether the **economy can DISCOVER better filters than a fixed bank
while staying true to SPROUT**.

## Thesis to test

> A weight-shared convolutional layer governed by **gradient-as-currency** can
> *learn and self-size* its filter bank — beating the Phase-1 fixed bank — while
> staying legible (inspectable filters you can watch being born, consolidate, and
> die) and changing **nothing** in the validated core.

## Faithfulness: the filter is the economic unit

SPROUT's economy is per-synapse: a wire owns one weight + one gradient meter, and
confidence/prune/grow read that wire's load and demand. Weight-sharing breaks
"one wire = one weight" — so we move the economic unit up one level: **a filter**.
A filter's taps *are* its weight-shared parameters; applying it densely across the
image *is* its placements. The gradient meter lives on the filter and **aggregates
the gradient over every placement** (which is exactly the conv weight gradient).
Everything else is the same economy, one level up:

| SPROUT (wire) | Conv-SPROUT (filter k, kernel θ_k) |
|---|---|
| weight `w` | kernel `θ_k` (kh·kw taps, shared across all positions) |
| `M=EMA|g|`, `S=EMA g` | `M_k=EMA‖g_k‖`, `Svec_k=EMA g_k`; consistency `κ_k=‖Svec_k‖/(M_k+ε)` |
| load `=|w|/w̄` | `load_k=‖θ_k‖/mean_j‖θ_j‖` |
| demand `=M/M̄` | `demand_k=M_k/mean_j M_j` |
| 2D confidence `gain·imp·settled` | identical, with `imp=max(load_k−1,0)`, `settled=settledness(demand_k)` (reuse `currency.settledness`) |
| gated update `η/(1+c)` | identical per filter (reuse the `apply_gated_update` rule) |
| prune wire if `U=load+λ·demand<floor`, age>grace | prune filter k likewise (keep ≥ `K_min`) |
| grow ghost wire of max virtual gradient at a plateau | birth a filter at a plateau (split highest-**demand** filter, or random); keep ≤ `K_max` |

Why pruning works end-to-end (the economy closing the loop): if a filter is
useless, the **head's own economy** prunes the wires that read it → no gradient
flows back to the filter → its `M_k`→0 → `demand_k`→0 → the filter is pruned. The
two economies cooperate. Newborn filters are protected exactly as newborn wires
are: low load but **high demand** (the loss pushes them hard) ⇒ high prune utility.

## Architecture

```
image (H,W) → ConvEconomy(K filters, 3×3, valid) → ReLU → 2×2 max-pool → flatten
            → [the EXISTING sparse phasic-startle-k4 head Network] → softmax
```

- **One** conv layer, single input channel (grayscale). Depth/stacking is a later
  extension; v0 keeps scope sane.
- **Joint training, simultaneous wake:** every step, conv-forward → head-forward →
  head-backward → backprop through flatten/unpool/ReLU into the filters → update
  **both** economies' meters + confidence + gated weights.
- **Phasic structure at a shared settledness plateau** (one `SettlednessDetector`
  on the joint loss): the head rewires via the existing
  `currency.prune_currency`/`grow_currency`/`batch_edge_scores` (run on a batch of
  *current* conv features); the conv layer prunes low-utility filters and births
  new ones. Then re-settle. Mirrors the wake/sleep/startle rhythm.

## Code (all additive — nothing in the validated core changes)

| File | Responsibility |
|---|---|
| `sprout/conv.py` *(new)* | Pure vectorized `conv_valid_forward`/`backward`, `maxpool_forward`/`backward` (NumPy); `ConvEconomy` (filters + per-filter currency: meters, load/demand, 2D confidence, gated update, filter prune/grow). Reuses `currency.settledness`. |
| `sprout/conv_train.py` *(new)* | `ConvModel` (ConvEconomy + head `Network`); `ConvTrainer` (simultaneous wake step; periodic phasic rewire of both at a `sleep.SettlednessDetector` plateau; head input-layer delta to bridge into conv). Reuses `learning`, `currency`, `sleep`. |
| `conv_experiment.py` *(new)* | Multi-seed runner; writes a committed `docs/eval-runs/<name>/` folder: README (mean±std table + simple seed-bootstrap verdict), accuracy-curve PNG, **filter-visualization PNG** (legibility), metrics.json. |
| `tests/test_conv.py`, `tests/test_conv_train.py` *(new)* | TDD incl. the **numerical gradient check** (analytic conv backward vs finite differences) — the linchpin correctness guard. |

## Backprop chain (one sample)

forward: `preact_k = conv_valid(img, θ_k)` → `relu_k` → `(pooled_k, argmax_k)` →
flatten `feat` → `head.forward(feat)`.
backward: `head.backward(y)` ⇒ delta for non-input; **input-layer delta**
`d_feat[p] = Σ_post w(p,post)·delta[post]` → reshape to `d_pooled_k` → unpool
(scatter to argmax) → `·(preact_k>0)` → `g_k = Σ_positions d_preact_k[i,j]·img[i:i+kh, j:j+kw]`
(the placement-aggregated shared-weight gradient).

## Experiments

- **E1 — learned vs fixed (fixed K).** Arms (same head, same data/steps, ≥3 seeds):
  `fixed-hand` (Phase-1 bank, not learned) vs `learned-gated` at K=6 and K=12, on
  14×14. Cross-reference Phase-1 raw (0.892) / fixed (0.929). *Does the economy
  learn filters that beat the fixed bank?* (Also a plain-SGD-learned ablation to
  separate "learning helps" from "currency helps", budget permitting.)
- **E2 — self-sizing economy.** Start K0, enable phasic filter grow/prune (cap
  `K_max`). *Does it find a good K, beat fixed-K, and grow legible filters?* Report
  final K trajectory + filter visualizations.
- **E3 — only if E1/E2 warrant.** 784, birth-mode (split vs random), or depth.

## Success / honesty criteria

- **Primary:** `learned-gated` test acc clearly ≥ `fixed-hand` (seed-bootstrap),
  ideally past Phase 1's +4pt. If learned only ties fixed, that is an honest
  negative (the economy isn't finding better filters) and is reported as such.
- **Legibility:** filter visualizations show interpretable detectors; births/deaths
  tracked. A win that isn't legible is only half the thesis.
- **Compute:** report edge-steps / wall time; conv is cheap NumPy, the python-loop
  head dominates, so cost ≈ Phase 1.
- **Safety:** full suite + validate.py must stay green (additive code only).

## Risks

- **Gradient bugs** — mitigated by the numerical gradient check before any training
  run; build in stages (pure conv math → economy → trainer → experiments).
- **Filter pruning never triggers** (random filters keep their init norm) —
  mitigated by demand-driven utility + the head cutting off useless filters;
  fallback knob: small filter weight-decay so unused filters fade (load→0).
- **Birth disrupts the function** — accepted: births fire only at a plateau and we
  re-settle (the phasic philosophy); split-highest-demand is gentler than random.
- **Scope creep** — one conv layer, grayscale, no stacking in v0.
```
