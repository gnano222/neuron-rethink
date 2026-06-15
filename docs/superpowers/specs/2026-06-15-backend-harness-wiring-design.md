# Wiring the vectorized backend into the eval harness (`--backend`)

**Date:** 2026-06-15 · **Branch:** sprout-v1 · **Status:** approved (phase 2 of the
vectorized-backend spec)

## Goal

Let the eval harness (`evaluate.py` / `evals.runner`) train on the fast
`ArrayNet` path via a `--backend array` flag, so the full comparison machinery
(multi-seed, bootstrap verdicts, published charts/README) can run on big nets that
are impractical on the object Trainer. The object backend stays the **default and
the validated reference**.

## Faithfulness contract (decided)

The array path is **statistically equivalent but not identical** to the object
path over long runs (float-drift in scatter-sums + burst-timing RNG desync —
measured ~0.3–1.5pt on MNIST). Therefore:
- `backend="object"` is the **default**; all existing published runs and
  `validate.py` are unchanged.
- `backend="array"` is an opt-in **faster path**; **A/B within a backend**, not
  across. The eval **cache key includes `backend`** so array/object runs never
  collide.
- Correctness is enforced by **parity-with-tolerance**, not equality.

## Architecture

### `sprout/fast.py`: `ArrayTrainer` (mirrors the Trainer interface `run_one` uses)
A thin adapter exposing exactly what the harness loop/metrics read:
- `__init__(cfg, net, X, y, seed)` — builds `ArrayNet.from_network(net)`, a
  `SettlednessDetector` (same floor as Trainer: `startle_floor` or `½ln(K)`),
  `events=[]`, `step_idx=0`, holds `net`, `X`, `y`.
- `step(record=False)` — one vectorized wake step (`ArrayNet.step`); feed loss to
  the detector; on a settled phasic plateau run the burst: `sync_into(net)` → reuse
  `currency.prune_currency` / `grow_currency` (exactly `_rewire_phasic`) → **log
  `sleep`/`prune`/`grow` events** (so `final_snapshot`'s structural metrics work) →
  `ArrayNet.from_network(net)` rebuild → `detector.reset()`. Increment `step_idx`.
  Returns loss. (Single-sample; continuous-path `phasic_structure=False` also
  supported by mirroring `_rewire_currency`'s sleep-gated prune/grow.)
- `sync_into(net)` — delegate to `ArrayNet.sync_into` (write arrays → object net).
- `.events`, `.step_idx`, `.X`, `.y` attributes (X/y reassignable for the binary
  label-swap shift, like `Trainer`).

`train_array` is refactored to a thin wrapper over `ArrayTrainer` (one source of
truth for the burst logic).

### `evals/runner.py`
- `run_one`: choose `ArrayTrainer` when `spec.backend == "array"` else `Trainer`.
- The loop is unchanged except: when array, call `tr.sync_into(net)` **before**
  each `_snapshot` so `accuracy(net,…)` / `synapse_count` / per-synapse confidence
  read current state. (`len(net.synapses)` per step is already correct — it only
  changes at bursts, which mutate `net`.) `final_snapshot` runs on the synced
  `net` + `tr.events`, unchanged.
- `_cache_key` gains `"backend": spec.backend`.
- Array backend is **single-regime only**; `run_one_continual` stays object-only
  (raise a clear error if `backend="array"` with `regime="continual"`).

### `evals/spec.py` / `evals/cli.py`
- `SuiteSpec.backend: str = "object"`; `--backend {object,array}` (default
  `object`); threaded through `build_spec`.

## Testing

- **Cache key:** `_cache_key` differs for object vs array (same variant/seed).
- **Interface:** `ArrayTrainer` exposes `step`/`step_idx`/`events`/`sync_into`;
  `events` accrue `prune`/`grow`/`sleep` on a run that bursts.
- **Harness smoke:** `run_one(backend="array")` on a small mnist14/spirals spec
  returns a well-formed RunResult (series populated, `final_test_acc` in [0,1],
  `n_neurons` right).
- **Parity (tolerance):** on a tiny single-task spec, `run_one` object vs array
  give `final_test_acc` within a loose tolerance (e.g. 0.1) — equivalence, not
  equality.
- **Regression:** full existing suite green (object default untouched);
  `train_array` still passes via the `ArrayTrainer` refactor; `validate.py` 7/7.

## Supported configs on the array backend

- **Phasic** (`phasic_structure=True`) — full burst support (the promoted default
  and every sweep arm).
- **Static** (no plasticity, e.g. `fully-connected`) — pure wake, never bursts;
  works trivially.
- **Continuous-with-plasticity** (`phasic_structure=False` AND
  `enable_prune/grow`) — **NOT supported**: `ArrayTrainer.__init__` raises
  `NotImplementedError` (use `backend="object"`). Reimplementing the legacy
  continuous `_rewire_currency` faithfully isn't worth it; those variants
  (`currency`, `sleep`) are A/B references that stay on the object path.

## Out of scope

Continual regime on the array path (object-only); continuous-with-plasticity
variants (raise); lazy-meter / tugofwar on the array path (already raised by
`ArrayNet`); changing the default backend; any new dataset.
