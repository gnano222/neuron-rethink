# SPROUT comparative evaluation harness — design

**Date:** 2026-05-30
**Branch:** sprout-v1
**Status:** approved, ready for implementation

## Goal

A reusable harness that runs a set of **variants** across **N seeds in parallel**,
tracks a fixed metric schema **over training time**, aggregates across seeds with
**uncertainty + a significance verdict vs a baseline**, and emits a **scorecard**
(table + CSV/JSON) plus diagnostic plots.

It serves two jobs at once:
- **(a) research scorecard** — "did variant X actually beat the baseline?" with
  multi-seed confidence, not single-run noise.
- **(b) mechanism diagnostics** — rich over-time traces of synapse dynamics and
  quality, to understand *why* a variant behaves as it does.

This **replaces** the hardcoded single-seed `compare.py`, leaves `validate.py`
(single-run pass/fail) untouched, and changes **nothing** in `sprout/`.

### Approved decisions

1. **Baseline:** `legacy-full` (the README's tuned baseline). Verdicts read
   "did this variant catch up to / pull ahead of the tuned v1 system?"
2. **Default seeds:** 5 (fast — ~1–2 min for the default suite).
3. **Held-out test set:** yes. Accuracy/loss reported on a held-out set from the
   same generator (different seed). Train accuracy kept alongside for continuity.
4. **`compare.py`:** retired (deleted). The harness subsumes it.

## Why no core change is needed (Approach C, in practice)

`trainer.events` already records `{"step", "type", "edge"}` (network.py state,
populated in train.py), and `Synapse` already carries `weight`, `confidence`,
`grad_mag`, `grad_signed`, `age`. Every lifespan / oscillation / utility /
calibration metric is therefore computable from **(final net) + (event log) +
(a fresh gradient pass over the test set)**. The harness owns its own minimal
training loop (mirroring `run.run` without rendering) and snapshots metrics at
record steps. No hook inside `Trainer` is required.

## Module layout (new isolated `evals/` package)

Named `evals/` (not `eval/`) to avoid shadowing Python's builtin `eval()`.

```
evals/
  __init__.py
  spec.py        # Variant registry (name -> Config factory); SuiteSpec dataclass
  runner.py      # run_one(variant, seed, suite) -> RunResult; ProcessPool; disk cache
  metrics.py     # PURE functions: timepoint snapshot + final snapshot -> metric dicts
  aggregate.py   # across seeds: mean/std/95% CI; bootstrap diff-vs-baseline + verdict
  report.py      # scorecard.md + scorecard.csv + stdout table + plots
  cli.py         # argparse entry
evaluate.py      # thin top-level entry -> evals.cli.main()
```

Reuses `sprout.network.{build_graph, init_weights}`, `sprout.train.{Config,
Trainer, accuracy}`, `sprout.data.{generate_spirals, generate_blobs}`.
`RunResult` is plain dicts/lists (picklable → safe to return from worker
processes), cached to `output/eval/cache/<config-hash>.json`.

## Variant registry (`spec.py`)

A variant maps a name → a `Config`, with the tuned spirals hyperparameters baked
in (lifted verbatim from the proven `compare.py` / `validate.py` configs — the
eval must use the *tuned* settings, which `run.PRESETS` alone does not carry):

- **`legacy-full`** (baseline): `Config(eta_base=0.02, enable_eligibility=True,
  enable_confidence=True, enable_prune=True, enable_grow=True, theta_prune=0.001,
  prune_warmup=6000)`
- **`currency`** (current default): `Config(eta_base=0.02, grad_currency=True,
  enable_confidence=True, enable_prune=True, enable_grow=True, gamma_dec=0.001,
  t_struct=200)`
- **`currency-grace`**: as `currency` + `t_grace=1000, grow_bar_frac=2.0`.
- **`core`**: `Config(eta_base=0.02)` (plain sparse SGD; plasticity metrics
  report N/A where inapplicable).

The registry is a `dict[str, Callable[[], Config]]` so future variants
(e.g. hysteresis) are one entry. `SuiteSpec` holds: `variants`, `seeds`,
`dataset`, `steps`, `shift_steps`, `record_every`, `baseline`, `layers`,
`density`, `n_points`, `turns`, `noise`, `test_seed_offset`.

## Run model & instrumentation

`run_one(variant_name, seed, suite) -> RunResult`:

1. Build train set `generate_<dataset>(n=n_points, seed=seed, ...)` and a
   **held-out test set** with `seed = seed + test_seed_offset` (default offset
   10000), same distribution params.
2. `build_graph(layers, density, seed)`, `init_weights(seed)`, `Trainer(cfg, ...,
   seed)`. Capture `initial_edges = set(net.synapses)` (for lifespan
   reconstruction).
3. Train `steps` steps. At each record step, compute a **timepoint snapshot** and
   append to series.
4. If `shift_steps > 0`: swap labels (`tr.y = 1 - y_train`; test eval uses
   `1 - y_test`), continue `shift_steps` steps recording the same series, mark
   `shift_start_index`.
5. At the end, compute the **final snapshot** (distributions + event-derived
   metrics + a fresh-gradient pass).

`RunResult` fields: `variant`, `seed`, `config` (dict), `series` (dict of lists,
incl. `rec_step`), `final` (flat dict of scalar metrics), `shift_start_index`,
`initial_edges`, `events` (the raw log).

### Timepoint snapshot (cheap; every record step)

`test_accuracy`, `test_loss`, `train_accuracy`, `train_loss`, `synapse_count`,
`cum_grow`, `cum_prune` (cumulative event counts ≤ this step), `mean_confidence`,
`mean_utility` (using metered `grad_mag`), `dead_unit_count`.

### Fresh-demand pass (fairness across architectures, at final snapshot only)

Legacy variants never populate `grad_mag` (only the currency path meters it), so
utility/calibration must not depend on the metered EMA when comparing across
architectures. At the final snapshot, do **one** full-test-set gradient
accumulation pass (the `validate.py` `_currency_meter_fidelity` pattern):
`fresh_demand[edge] = mean_over_test |dL/dw|`. Utility, calibration, and
meter-fidelity all use `fresh_demand`, making them comparable for every variant.

## Metric schema — 4 families + performance/efficacy

Two granularities: **series** (per record step → curves) and **final** (scalars
for the scorecard). Each scalar metric is tagged `higher`/`lower`/`neutral` for
verdict direction.

**A. Prediction performance**
- series: `test_accuracy`, `test_loss`, `train_accuracy`, `train_loss`.
- final: `final_test_acc`, `max_test_acc`, `final_train_acc`; with shift:
  `pre_shift_test_acc`, `recovered_test_acc`.

**B. Training efficacy**
- `steps_to_90`, `steps_to_95` (first `rec_step` reaching threshold; `inf` if
  never) — lower better.
- `auc_test_acc` = `trapz(test_acc, rec_step) / (rec_step[-1]-rec_step[0])` —
  mean accuracy over training; higher better.
- `final_acc_stability` = std of `test_accuracy` over last K=10 records — lower
  better.
- with shift: `recovery_steps` (post-shift steps to regain `pre_shift_test_acc`;
  `inf` if never) — lower better; `recovery_gap` = `pre_shift - recovered` —
  lower better.

**C. Synapse structure**
- `synapse_count_start/peak/end`; `n_grow_events`, `n_prune_events`;
  `turnover` = `(n_grow+n_prune)/mean(synapse_count)` — lower better;
  `max_grows_into_one_neuron` (churn, from events) — lower better;
  `mean_fan_in`, `mean_fan_out` (end state); `effective_density` (live edges /
  fully-connected edge count).

**D. Synapse quality**
- *Utility/value*: per-survivor `u = |w|/mean|w| + λ·fresh_demand/mean(fresh_demand)`
  (λ = `lam_prune`, default 1.0). Report `mean_utility`, `p10_utility`,
  `freeloader_frac` (`u < prune_u_floor`) — freeloader_frac lower better.
- *Stability/lifespan*: `mean_survivor_age`, `median_survivor_age` (higher =
  more committed); `mean_pruned_lifespan` (steps a pruned wire lived, from
  events; birth = last grow before prune, else 0 for `initial_edges`);
  `oscillation_frac` = (#edges grown ≥2×)/(#distinct grown edges) — lower
  better; `max_regrow` = max over edges of `grow_count-1` — lower better.
- *Confidence calibration*: `conf_utility_corr` = corr(confidence, utility) over
  survivors — higher better; `frozen_freeloader_frac` (confidence > 1 AND
  `u < prune_u_floor`) — lower better; `high_conf_survival` (fraction of wires
  with confidence > 1 that survive to end).
- *Effective capacity*: `dead_unit_count` (hidden neurons with max test-set
  activation < ε) — lower better; `inert_synapse_frac` (`|w| < ε`) — lower
  better; `used_vs_allocated` = live edges / `initial_edges` count.

**Sanity (currency only):** `meter_fidelity` = corr(`grad_mag`, `fresh_demand`)
over survivors — higher better (reused from `validate.py`).

Metrics that don't apply to a variant (e.g. confidence for `core`) report `None`
/ `NaN` and render as `—`.

## Aggregation & significance (`aggregate.py`)

For each (variant, metric): `mean`, `std`, 95% CI across seeds.

Against the baseline variant, for each other variant & metric: bootstrap the
mean-difference distribution by resampling seeds with replacement (default
10000 resamples, seeded RNG for reproducibility); 95% CI = the [2.5, 97.5]
percentiles of `mean(variant*) - mean(baseline*)`. Verdict, using each metric's
direction tag:
- CI entirely on the better side of 0 → **better**
- CI entirely on the worse side → **worse**
- CI straddles 0 → **≈ (no clear difference)**

Pure-NumPy bootstrap → **no new dependency** (scipy is not installed). `inf`
values (e.g. never reached threshold) are handled: if any seed is `inf`, report
the metric as `inf`/`—` and skip the bootstrap for it.

## Outputs (`report.py`)

Written under `output/eval/<suite-name>/`:
- `scorecard.md` and `scorecard.csv`: rows = metrics grouped by family, columns =
  variants showing `mean ± std`, with a vs-baseline marker (`▲`/`▼`/`≈`) on
  non-baseline columns.
- `metrics.json`: full aggregates + per-seed raw values.
- stdout: the scorecard table (compare.py style, but `mean ± std` + verdicts).
- plots (matplotlib, already a dep):
  - `acc_curves.png` — test-accuracy curves, per-seed mean ± band, shift line.
  - `count_curves.png` — synapse-count mean ± band.
  - `churn_curves.png` — cumulative grow/prune (turnover proxy) over time.
  - `quality_<variant>.png` — utility histogram, confidence-vs-utility scatter,
    survivor-age histogram (one figure per variant).
  - `verdict_heatmap.png` — variant × metric verdict grid.

## CLI (`cli.py` / `evaluate.py`)

```
python evaluate.py --variants currency,legacy-full,core --seeds 10 \
  --dataset spirals --steps 30000 --shift 6000 --baseline legacy-full \
  --jobs 8 --out output/eval/run1
```

Defaults: `--variants currency,legacy-full`, `--seeds 5`, `--dataset spirals`,
`--steps 30000`, `--shift 0`, `--baseline legacy-full`, `--jobs` = `os.cpu_count()`,
`--record-every 200`, `--no-cache` off. `--out` defaults to
`output/eval/<dataset>_<timestamp>`.

## Parallelism & caching

- `ProcessPoolExecutor(max_workers=jobs)` over the cartesian product of
  (variant, seed). Each job is independent (own seed; no shared state). Results
  are deterministic per seed regardless of `--jobs`.
- The hand-rolled forward/backward are Python scalar loops over adjacency lists,
  not large BLAS calls, so thread oversubscription is essentially a non-issue;
  still, set `OMP_NUM_THREADS=1` (and siblings) at process start defensively.
- Cache key = stable hash of `(variant config dict, seed, dataset, steps,
  shift_steps, record_every, layers, density, n_points, turns, noise,
  test_seed_offset)`. Cached `RunResult` JSON lives in `output/eval/cache/`.
  `--no-cache` forces recompute. Adding a variant / re-plotting reuses cache.

## Testing (TDD, matching the 88-test suite culture)

- `tests/test_eval_metrics.py` — pure functions on tiny hand-built nets +
  synthetic event logs with known answers. E.g. events `[grow E@100, prune
  E@150, grow E@300]` → `max_regrow = 1`, first-instance `pruned_lifespan = 50`;
  a 2-neuron net with known weights → known utility; a permanently-silent neuron
  → `dead_unit_count` counts it.
- `tests/test_eval_aggregate.py` — seeded bootstrap on known arrays (CI brackets
  the true difference); verdict direction correct for higher- and lower-better
  metrics; `inf` handling.
- `tests/test_eval_runner.py` — a 300-step smoke run returns a well-formed
  `RunResult` with every schema key present; determinism (same seed → identical
  `final`); cache round-trip (second call hits cache, identical result).
- `tests/test_eval_report.py` — scorecard builds from a synthetic aggregate dict
  without error; CSV has the expected columns; markdown is non-empty.
- Fast end-to-end in `test_eval_runner.py`: 2 variants × 2 seeds × 300 steps
  completes in a few seconds and produces a scorecard.

## Non-goals (YAGNI)

- No new model mechanisms (hysteresis etc. — separate work this harness will
  *measure*).
- No live / web dashboard — static plots + markdown.
- No ROC / calibration curves (2-class toy; accuracy + loss suffice).
- `validate.py` is untouched (different job: single-run pass/fail).

## File-by-file change list

- **add** `evals/__init__.py`, `evals/spec.py`, `evals/runner.py`,
  `evals/metrics.py`, `evals/aggregate.py`, `evals/report.py`, `evals/cli.py`.
- **add** `evaluate.py` (top-level entry).
- **add** `tests/test_eval_metrics.py`, `tests/test_eval_aggregate.py`,
  `tests/test_eval_runner.py`, `tests/test_eval_report.py`.
- **delete** `compare.py`.
- **edit** `README.md`: replace the `compare.py` references in Quick start /
  Code layout with `evaluate.py`; keep the honest-comparison table (regenerate
  later from the harness).
- **no change** to `sprout/`.
