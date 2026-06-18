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

## Follow-up: longer budget (60k steps) — the real bottleneck is STABILITY, not discovery

The experiments above used a short 15k-step budget. Re-running the key comparisons
at 60k steps (where the raw startle-k4 head reaches its 0.93 ceiling) on one
consistent ruler (`docs/eval-runs/conv-sprout-long-digits`, `…-long-motifs`):

| Task | Arm | final | peak (max) | curve shape |
|---|---|---|---|---|
| Digits | fixed-hand-k6 | **0.931** | 0.939 | smooth, monotone, stable |
| | learned-k6 | 0.889 (DOWN) | 0.928 | peaks ~0.905 then **drifts down** |
| | selfsize-2→12 | 0.915 (DOWN) | 0.933 | wobbly, ends below peak |
| Motif | fixed-hand-k6 | 0.425 | 0.453 | steady |
| | learned-k6 | 0.453 (~) | **0.507** | wobbly, higher peaks |
| | selfsize-2→12 | 0.414 (~) | **0.522** | very wobbly (one −0.11 crash), highest peak |

Two things change with the real budget:

1. **Fixed conv reaches the 0.93 ceiling** (0.931), confirming the architecture is
   sound and the 15k numbers were just under-trained.
2. **Learned/self-sizing filters reach competitive-or-higher PEAKS but do not hold
   them** — their curves wobble ±2–5pt and end below their peak, with high seed
   variance. On digits this makes fixed clearly better on final accuracy; on motifs
   the clear 15k win erodes to a final tie even though the learned PEAKS
   (0.51–0.52) plainly beat fixed (0.45).

So the limiter is **not filter discovery** — the peaks prove the economy finds good
(even better) filters. The limiter is **consolidation/stability**: the filters keep
moving, so the head trains on a shifting representation (a moving target), and the
self-sizing births/deaths add discrete shocks (the −0.11 motif crash). Fixed filters
sidestep this entirely by being frozen. SPROUT already solved exactly this for
*wires* (phasic consolidation + confidence-freezing settled wires); the filter
economy needs the same — freeze/decay settled filters so the representation settles.
The per-filter confidence gating is present but evidently too weak to stop late
drift. **This is the concrete next lever.**

## Consolidation closes the gap — and FLIPS the verdict

Acting on the diagnosis above, I added **filter-LR consolidation**: wind the filter
learning rate down to ~0 over training (cosine schedule) so filters settle near
their peak and the head finishes on a stationary representation — the SPROUT
"consolidate so downstream can settle" idea, applied to filters. Re-running at 60k
(`docs/eval-runs/conv-sprout-consolidate-{digits,motifs}`):

| Task | Arm | final | peak | late drift | verdict vs fixed |
|---|---|---|---|---|---|
| Digits | fixed-hand-k6 | 0.931 | 0.932 | 0.001 | — |
| | learned-k6 (no consol.) | 0.889 | 0.905 | 0.015 | DOWN |
| | **learned-k6 + cosine** | **0.953** | 0.953 | **0.000** | **UP (+2.2pt)** |
| | self-size + cosine | 0.940 | 0.947 | 0.007 | ~ |
| Motif | fixed-hand-k6 | 0.425 | 0.432 | 0.007 | — |
| | learned-k6 (no consol.) | 0.453 | 0.472 | 0.019 | ~ |
| | **learned-k6 + cosine** | **0.696** | 0.696 | **0.000** | **UP (+27pt)** |
| | **self-size + cosine** | **0.752** | 0.752 | **0.000** | **UP (+33pt)** |

Consolidation did two things at once:

1. **Killed the instability** — late drift collapsed to ~0; final now tracks the
   peak. The diagnosis was correct: the limiter was stability, not discovery.
2. **Unlocked large gains** — with a *stationary* learned representation the head
   can fully exploit it. On digits the learned-and-consolidated filters now **beat
   the fixed hand bank (0.953 vs 0.931)** and clear the old ~0.93 ceiling; on the
   filter-sensitive motif task they jump to **0.70–0.75 vs fixed 0.45** (+25–33pt).
   The motif curves even *surge* in the final third as the filters lock in.

**This reverses the earlier digit negative.** The 15k/60k "learned ties-or-loses
fixed" result was an artifact of un-consolidated filter drift. Once the filters
consolidate, **learning beats fixed on digits too**, and self-sizing + consolidation
is now stable and strong. Phase 2 works: the economy learns *and* consolidates its
own filters, beating a fixed bank on both ordinary (digits) and filter-sensitive
(motif) tasks. Recommended setting for learned conv: **cosine filter-LR wind-down**
(`conv_eta_schedule="cosine"`).

## Practical recommendation (updated after consolidation)

Use the **learned conv economy WITH filter-LR consolidation** (`conv_eta_schedule=
"cosine"`) — given a real training budget it beats the fixed hand bank on digits
(0.953 vs 0.931, above the old ~0.93 ceiling) and dominates on filter-sensitive
tasks (+25–33pt). The fixed Phase-1 bank remains the right *cheap* choice when the
budget is short (≤15k steps, where nothing has converged) or when you can't afford
the extra filter-learning compute. The earlier "just use the fixed bank" advice held
only because un-consolidated filters were unstable; consolidation removes that
caveat.

## Redundancy pruning: lets self-sizing trim, but the trim is gentle

The remaining gap was that self-sizing only *grew* — pruning detected inertness, not
redundancy. Added two redundancy signals that fire at sleep plateaus and drop the
lower-utility member of any over-correlated pair (start full at 12, no growth):

| Task | signal | filters end | accuracy | vs no-prune control |
|---|---|---|---|---|
| Digits | (no prune control) | 12.0 | 0.955 | — |
| | kernel cosine > 0.85 | 11.4 | 0.956 | held |
| | activation corr > 0.9 | 9.6 | 0.954 | held |
| | **both (kernel ∪ activation)** | **9.6** | 0.954 | held |
| Motif | (no prune control) | 12.0 | 0.743 | — |
| | kernel cosine > 0.85 | 10.4 | 0.736 | held |
| | activation corr > 0.9 | 11.8 | 0.736 | held |
| | **both (kernel ∪ activation)** | **10.4** | 0.728 | held |

The two signals are **complementary, not nested** (an initial guess that functional
subsumes kernel held on digits but failed on motifs): functional trims more on
digits (9.6 vs 11.4), kernel trims more on motifs (10.4 vs 11.8). **"both" is the
union and robustly gets the better of the two per task** (digits 9.6, motifs 10.4),
at held accuracy — so it's the sensible default (no need to pick a signal). It does
not, however, escape the ceiling below: combining trims a bit more reliably but not
dramatically more.

Both signals **work and never cost accuracy** (the pruned filters were genuinely
redundant), so it's a *free* efficiency trim — on digits the **functional**
(activation-correlation) signal removes 12→9.6 filters (~20% fewer) at equal
accuracy, and beats the kernel-shape signal (which is too strict).

At the conservative 0.9 cutoff the trim looked gentle (12→9.6), but a **cutoff sweep
on the functional signal** (`docs/eval-runs/conv-sprout-actsweep-digits`, digits 60k,
baseline = the 12-filter no-prune control) shows the cutoff is a clean dial and the
gentleness was just a too-high threshold:

| activation-corr cutoff | filters end | final acc | vs no-prune |
|---|---|---|---|
| none (control) | 12.0 | 0.955 | — |
| 0.90 | 9.6 | 0.954 | ~ (free) |
| **0.80** | **6.8** | **0.951** | **−0.4pt (sweet spot)** |
| 0.70 | 3.8 | 0.937 | −1.8pt |
| 0.60 | 3.0 | 0.935 | −2.0pt |
| 0.50 | 3.0 | 0.932 | −2.3pt |

- **The cutoff cleanly trades filters for accuracy** — monotonic and predictable.
- **0.80 is the sweet spot:** it leans the bank from 12 down to **~7 filters
  (matching the hand-bank's 6) at essentially no cost** (−0.4pt). So self-sizing
  *can* find a lean bank after all — the "gentle trim" was an artifact of the 0.9
  threshold, not a real ceiling.
- **Below 0.7 it over-prunes** — cutting genuinely-useful filters, ~−2pt for 3–4.
- **Pruning does not *improve* accuracy** (no regularization bonus here): best is the
  full 12-filter bank; pruning trades a little accuracy for far fewer filters.
- **At matched filter count the self-sized learned bank beats fixed:** ~7 learned-
  and-consolidated filters at 0.80 score **0.951 vs the fixed hand-k6's 0.931** (+2pt).

Recommended redundancy setting: **functional (activation-correlation) at ~0.80** —
the knee of the curve. (Gradient-trained filters still diversify, so you can't trim
to the bare minimum without cost, but ~7 is free.)
