# Conv-SPROUT (Phase 2): findings

**Date:** 2026-06-16 · branch sprout-v1 · spec
`docs/superpowers/specs/2026-06-16-conv-sprout-phase2-design.md`

## What was built

A faithful realization of weight-sharing **inside** the gradient-as-currency
economy: the economic unit moves up from the wire to the **filter**. A filter's
taps are its weight-shared parameters, dense application is its placements, and a
single gradient meter on the filter aggregates the gradient over every placement
(== the conv weight gradient). The same currency readouts run one level up — 2D
confidence, demand-driven pruning, plateau-gated growth, `η/(1+c)` gated learning.
A `ConvEconomy` front-end feeds the **existing** sparse phasic head, trained jointly.

- New, **additive** modules only: `sprout/conv.py` (gradient-checked conv/pool +
  `ConvEconomy`), `sprout/conv_train.py` (`ConvModel` + `ConvTrainer`),
  `conv_experiment.py`. The validated core (network/currency/fast/validate.py) is
  untouched. Two numerical gradient checks pass (conv alone; the head→conv bridge).
- 332 tests pass; validate.py still 7/7.

## The question

Phase 1 showed *fixed* translation-invariant features beat raw pixels, and that
**hand ≈ random** filters — i.e. filter *quality* didn't matter, only the conv
*architecture* (translation-invariance + pooling). Phase 2 asks: can the economy
**discover better filters than a fixed bank**, and **self-size** its bank, while
staying legible?

## Results (5 seeds, 15k steps; verdict = 95% seed-bootstrap CI vs the fixed bank)

| Experiment | Arm | test acc | vs fixed | note |
|---|---|---|---|---|
| **E1** learned vs fixed (`…e1-learned-vs-fixed`) | fixed-hand-k6 | 0.897 | — | |
| | learned-k6 | 0.894 | ~ | learns legible edge detectors, no gain |
| | learned-k12 | 0.888 | ~ | |
| **E2** self-sizing (`…e2-selfsizing`) | selfsize-2→12 (split) | 0.872 | DOWN | grows 2→9, over-grows |
| | selfsize-2→12 (random) | 0.881 | ~ | grows 2→8 |
| | selfsize-12→12 (prune) | 0.888 | ~ | **stayed 12** — prune never fired |
| **E3** tight budget (`…e3-tight-k2`,`-k3`) | learned-k2 | 0.860 | ~ | |
| | learned-k3 | 0.855 | DOWN | high seed variance (±0.045) |
| **E4** motif positive control (`…e4-motifs`) | fixed-hand-k6 | 0.371 | — | chance = 0.167 |
| | **learned-k6** | **0.436** | **UP (+6.5pt)** | learns matched filters |
| | **selfsize-2→12 (random)** | **0.460** | **UP (+8.9pt)** | grows 2→10.8 |

## Verdict: the mechanism works; MNIST is the wrong testbed

- **On MNIST the economy ties or loses to a fixed bank at every budget tested
  (k = 2, 3, 6, 12).** This is not a mechanism failure — it is the *data*: MNIST is
  filter-insensitive (Phase 1 already showed hand ≈ random). The learned filters
  are legible oriented edge/gradient detectors — the same kind the hand bank
  supplies — so there is no quality headroom to win, and learning merely adds
  seed variance (random-init local optima).
- **On a task where filter quality IS the bottleneck (E4 matched-filter motifs),
  the economy clearly wins:** learned +6.5pt and self-sizing +8.9pt over the fixed
  bank, with the learned kernels specializing into the task's motifs. This proves
  the mechanism is sound (no bug suppressing learning) and that the MNIST null is a
  property of the data, not the method.
- **Self-sizing grows but does not lean out.** The grow economy fires (2→8–11
  filters at plateaus) and helps on E4, but **filter pruning never triggers**: with
  all filters read by the head, every filter keeps nonzero demand, so
  `U = load + λ·demand ≈ 2 ≫ floor`. The prune signal detects *inertness*, not
  *redundancy* — the open gap. A redundancy-aware consolidation signal
  (filter–filter correlation) is the natural next step, and a very SPROUT one
  (consolidating duplicate detectors, like sleep consolidation for wires).

## Practical recommendation

For MNIST-family data, use the **cheap Phase-1 fixed bank** (+3–4pt, near-zero
cost) — learning the filters adds compute and variance without accuracy. Reserve
the Phase-2 economy for tasks whose discriminative features are *specific local
patterns* (where E4-style gains appear). The faithful machinery is in place either
way, and the one concrete improvement worth building is redundancy-aware filter
consolidation so self-sizing can find a genuinely lean bank.
