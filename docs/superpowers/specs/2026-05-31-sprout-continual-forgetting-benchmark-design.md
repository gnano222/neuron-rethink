# SPROUT continual-learning (forgetting) benchmark — design

**Date:** 2026-05-31
**Branch:** sprout-v1
**Status:** approved, ready for implementation

## Goal

Add a **forgetting benchmark** to the eval harness: a continual-learning regime
where an old task **stays valid** while a new task is learned, so we can measure
whether a variant *retains* old knowledge (the thing confidence-freezing is
supposed to buy) and whether it can ultimately *hold both tasks at once*.

### Motivating finding (why this is needed)

The existing concept-shift regime is a **label swap** (`runner.py:85` —
`y_tr_sw = 1 - y_tr`). A label swap makes the old answer the *exact opposite* of
correct, so the only good behaviour is to **overwrite** the old solution fast.
`validate.py:226-258` even asserts confidence should *fall* after the swap. That
measures **re-adaptation**, which is the *opposite* of forgetting-resistance.
Therefore the current harness **cannot measure forgetting**, and tuning
confidence calibration against it would optimise the wrong direction.

Confidence's stated job is **resist forgetting** (protect already-learned wires).
To pursue that honestly we first need a benchmark that can *see* forgetting:
two tasks that remain *simultaneously satisfiable* by one network.

### Scope (explicit)

- **In scope:** the continual benchmark (offset-spiral tasks), the 3-phase
  protocol, dual-task tracking, forgetting/consolidation metrics, and their
  integration into the existing run / aggregate / report / publish pipeline.
- **Out of scope:** the confidence redesign itself (2D weight-aware confidence,
  sticky freeze). That is a **separate follow-on spec** that will *consume* this
  benchmark. Also out: per-wire "which task does this wire serve" attribution
  (a nice future calibration metric, not v1).

This **changes nothing in `sprout/`** except two backward-compatible optional
params on `generate_spirals` (`center`, `scale`, both defaulting to today's
behaviour). The proven single-task + label-swap path is left **untouched**.

## Core constraint that shapes the design

"Old task stays valid" *forces* the two tasks into **disjoint input regions**.
The net has one 2-way output: if A and B sat in the same place but wanted
different answers, being right on B necessarily means being wrong on A — they
cannot both hold. The only way both stay correct is separate regions of the
plane, so one network satisfies both. **Forgetting pressure** then comes from the
tiny net's *scarce shared hidden capacity*: both spirals must be encoded with the
same few units, so learning B competes for the wires A relied on.

**Crucial refinement found at implementation time (see Findings):** the regions
must be disjoint *without pushing either task off-centre*, because an off-centre
input cloud gives the tiny net a large DC bias that kills ~half its ReLU units
(capacity collapse → unlearnable). Lateral separation can't satisfy both. The
resolution is **concentric (radial) separation**: an inner annular spiral and a
disjoint outer one, **both centred on the origin** (zero-mean → learnable) yet in
separate radial bands (jointly valid).

## Approved decisions

1. **Task pair:** two **concentric spirals** — inner annular spiral (task A) and
   a disjoint outer one (task B), both origin-centred. *(Originally specced as
   lateral left/right offset spirals; changed to concentric after the off-centre
   capacity-collapse finding below.)*
2. **Protocol:** **A → B → A+B** (three phases). Phase A learns the inner spiral;
   phase B trains on the outer spiral *only* (A may erode); phase A+B trains on
   the interleaved union (consolidation — can it hold both?).
3. **Tracking:** record *both* spirals' held-out accuracy at every snapshot,
   across all phases, yielding the forgetting curve and consolidation curve.
4. **Integration:** an **isolated new regime** (`regime="continual"`) with its own
   run path; the single-task + label-swap path is unchanged.

## §1 Data — concentric (radial-band) spirals

Replace the spiral radius with an explicit band on `sprout/data.generate_spirals`
(defaulting to current behaviour, so existing tests/runs are unaffected):

- `r_lo=0.2`, `r_hi=1.0` — the radius sweeps `[r_lo, r_hi]` linearly with the
  spiral parameter. The default `(0.2, 1.0)` reproduces the original spiral
  byte-for-byte (pure arithmetic around unchanged RNG draws). *(The earlier
  `center`/`scale` params for lateral offset were removed — that geometry is
  unlearnable; see Findings.)*

Continual tasks (both origin-centred, disjoint by radius):
- **Task A (inner):** `r_lo=0.15, r_hi=0.55`
- **Task B (outer):** `r_lo=0.65, r_hi=1.05` (gap at ~0.6)
- **`continual_turns=0.6`** — gentler than the single-task `turns=1.0`, so the
  4-arm union stays learnable by the default net (see Findings: capacity, not
  off-centredness, limits the union; bigger nets don't help, gentler turns do).

**Sanity check (done):** with the default net, inner learns to ~1.0, outer to
~0.98, and the union to ~0.89 — enough consolidation headroom.

## §2 Sequencing — `run_one_continual(variant, seed, spec)`

A new run path in `evals/runner.py`, parallel to `run_one` (not a refactor of it):

- **Generate** Task A: `generate_spirals(r_lo=inner_r_lo, r_hi=inner_r_hi,
  seed=seed, turns=continual_turns, noise)`; held-out A-test at
  `seed + test_seed_offset`.
- **Generate** Task B: same with the outer band and an offset seed stream
  (`seed + 20000`) so B is not a mirror of A; B-test at
  `seed + 20000 + test_seed_offset`.
- **Phases** (one network, one Trainer, reusing `apply_gated_update` etc.):
  - Phase A — `steps_a`: `tr.X, tr.y = A_train`.
  - Phase B — `steps_b`: `tr.X, tr.y = B_train` (A may erode here).
  - Phase A+B — `steps_ab`: `tr.X, tr.y = concat(A_train, B_train)` (the Trainer
    samples uniformly at random, so the union is naturally interleaved).
- **Defaults:** `steps_a = 15000`, `steps_b = 15000`, `steps_ab = 10000`.
- Record phase-boundary indices for metrics and plotting.

## §3 Snapshot — track both tasks

A continual-specific snapshot helper (does **not** alter the single-task
`_snapshot`). At every record step it appends:

- `phase` — `"A"` / `"B"` / `"AB"` label for the snapshot
- `test_accuracy_A`, `test_accuracy_B`
- `test_accuracy` — accuracy over the **union** of the two held-out test sets
  (named `test_accuracy` so `final_snapshot`'s perf/structure/quality metrics
  reuse it unchanged); `test_loss` likewise on the union
- existing dynamics: `train_accuracy`/`train_loss` (current phase),
  `synapse_count`, `mean_confidence`, `cum_grow`, `cum_prune`

Continual `SERIES_KEYS` are separate from the single-task ones.

## §4 Metrics — `continual_metrics(series, phase_bounds)`

Pure function in `evals/metrics.py`, measured at phase boundaries so each stays
clean (forgetting is read at the A→B boundary, *before* consolidation):

| Metric | Definition | Direction |
|---|---|---|
| `a_peak` | `test_accuracy_A` at end of phase A | higher |
| `b_learned` | `test_accuracy_B` at end of phase B | higher |
| `forgetting` | `a_peak − test_accuracy_A(end of B)` | **lower** |
| `consolidation` | `min(test_accuracy_A, test_accuracy_B)` at end of A+B | higher |
| `relearn_gap` | `a_peak − test_accuracy_A(end of A+B)` | lower |

`min(A, B)` is the honest "can it hold *both*" measure (the weaker task gates the
score). Direction tags registered for the aggregate/verdict/bootstrap machinery.

## §5 Report / publish

- **Money chart:** A-test and B-test accuracy vs step, with phase boundaries
  marked — A visibly decays during phase B, then both climb in A+B.
- Scorecard table gains `forgetting`, `b_learned`, `consolidation`.
- Same `--publish --run-name` mobile workflow; the continual run emits its
  README with the chart inline (renders on GitHub mobile).

## §6 Variants & the benchmark's own validity check

Default continual comparison: `core` (plasticity off — should forget hardest),
`currency` (freezing on), `legacy-full`. **The benchmark earns trust only if it
shows a gap** (core forgets more than the freezing variants). If `currency` does
*not* beat `core` here, that is itself an honest finding: confidence is not
protecting anything, and the redesign is warranted / its target reframed.

## §7 CLI

`evals/cli.py` / `evaluate.py`:

- `--regime {single,continual}` (default `single`)
- `--steps-a`, `--steps-b`, `--steps-ab`
- `--continual-turns` (the inner/outer radial bands use spec defaults)

`runner._cache_key` includes `regime` and all continual params so cached
single-task runs are never confused with continual ones.

## Testing (TDD)

- **data:** `generate_spirals(r_lo, r_hi)` places points in the expected radial
  band; default call is byte-for-byte unchanged (backward compat).
- **runner:** `run_one_continual` runs phases in order, produces a series with
  `phase` labels and both `test_accuracy_A/B`, and records phase boundaries.
- **metrics:** `continual_metrics` computes `forgetting` / `consolidation` /
  `b_learned` correctly on synthetic series (e.g. A drops then partially recovers).
- **integration:** one tiny continual run (few steps, 1 seed) completes and
  aggregates without error.
- **regression:** existing full suite stays green — isolated path.

## Findings during implementation

1. **Lateral offset is unlearnable (root cause: dead ReLUs).** A centred spiral
   learns to 0.99; one pushed fully off-centre plateaus at ~0.66 *regardless of
   scale or steps* (isolated by varying one variable at a time). Off-centre input
   = large DC bias → ~half the ReLU units die → capacity halved. Disjoint lateral
   tasks are *always* off-centre for at least one task, so lateral separation is
   stuck between "disjoint but unlearnable" and "learnable but overlapping
   (invalid)." → switched to **concentric** (origin-centred, disjoint by radius).
2. **The union is capacity-/hardness-limited, not width-limited.** Two full
   concentric spirals at `turns=1.0` give a 4-arm pinwheel the net tops out at
   ~0.63 on — and a *bigger* net does **not** help (0.88 vs 0.89). Gentler turns
   do: `turns=0.6` → union ~0.89 while each task stays non-linear. So
   `continual_turns=0.6` with the **default** net (no size bump needed).
3. **The benchmark does not (yet) discriminate freeze from no-freeze.** Real
   5-/3-seed runs: `currency` forgetting ≈ `core` forgetting (≈0.24 vs 0.23),
   consolidation comparable. Confidence-freezing buys ~zero retention here —
   consistent with the hypothesis that confidence *releases under contention*
   (the very weakness that motivates the redesign). **Shipped as-is per user
   decision** ("concentric only"); we deliberately did *not* add a positive
   control, so "currency = core" cannot yet be distinguished from benchmark
   insensitivity. Resolving that is left to the confidence-redesign follow-on.

## Success criteria

1. Full test suite green (existing + new continual tests). ✅ 168 pass.
2. A real continual run produces the forgetting curve + scorecard. ✅ Tasks
   learn (a_peak ~1.0, b_learned ~0.98); ~0.24 forgetting; chart renders.
3. ~~The benchmark discriminates core vs currency~~ — **null result, reported
   honestly** (see Findings 3): no freeze-vs-no-freeze gap; sensitivity unproven
   pending a positive control.
