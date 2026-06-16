# Adapting SPROUT to 8×8 digit recognition (staged 1 → 3)

**Date:** 2026-06-13
**Status:** approved (staged 1→3, autonomous build)

## Goal & constraints

Apply the gradient-as-currency SPROUT architecture — phasic structure + sleep
consolidation + startle (the promoted defaults) — to a **10-class digit
recognition** task, keeping the project's standing target: **high accuracy with
compute efficiency** (a sparse self-rewiring net that beats a fully-connected
control on edge-steps / wall-time / active-edge fraction).

**Dataset: scikit-learn `load_digits`** — 8×8 images, **64 features, 1797
samples, 10 classes**. Bundled in scikit-learn (zero download), and small enough
to stay inside the current pure-Python per-synapse compute budget. Full MNIST
(28×28, 784 features) is a deliberate *future* step; this design builds the data
seam so MNIST drops in as a loader, but MNIST itself waits on a vectorized
forward/backward (it is ~40k+ edges in Python scalar loops = hours/run).

## Why the core network needs no math changes

Verified in code — the architecture is already K-class / arbitrary-size general:

- `Network.forward` softmax over the output layer and `Network.backward`
  (`δ = probs − onehot`) work for any number of outputs
  (`sprout/network.py:117-127`, `:144-146`).
- `build_graph(layer_sizes, density, seed)` and `init_weights` (He init by the
  *real* sparse fan-in) are general over layer sizes (`sprout/network.py:175-214`).
- The startle trouble-floor auto-derives as `½·ln(K)` from the output-layer size
  (`sprout/train.py:235-236`) — generalizes from 2 to 10 classes for free.
- Eval metrics (accuracy/AUC via `argmax`, edge-steps, active-edge fractions) are
  class-neutral (`evals/metrics.py final_snapshot`). The only 2D-specific code is
  `sprout/viz.py`'s decision-boundary meshgrid panel, which the eval harness does
  **not** use (it renders only for the `run.py` GIF). Out of scope here.

The work is entirely in the **data layer, network sizing, and the eval-harness
seams**, not the learning math.

## Stage 1 — smoke-test (`run_digits.py`, repo root)

Purpose: prove the architecture learns 10-class digits and fix sizing/steps
*before* touching the harness. One self-contained script (mirrors `validate.py` /
`evaluate.py` placement; no viz, no GIF — just train + report).

- **Load:** `sklearn.datasets.load_digits` → `X (1797, 64)` float, `y (1797,)`
  int in `0..9`.
- **Normalize:** per-feature standardize using **train statistics only** —
  `(X − μ)/(σ + 1e-8)`, μ,σ computed on the train split (no leakage; some corner
  pixels are constant ⇒ the `eps` guard avoids divide-by-zero). Keeps inputs
  ~unit and zero-mean, matching `data.py`'s "roughly unit scale so the tiny net
  trains sanely".
- **Split:** seeded stratified 80/20 via
  `train_test_split(stratify=y, random_state=seed)`.
- **Build:** layers `(64, 64, 32, 10)`, density `0.4`, seed; then `init_weights`.
- **Config:** the promoted defaults (`phasic_structure`, `enable_sleep`,
  `startle` all True; `enable_confidence/prune/grow` True; `grow_demand_k=4`;
  `eta_base=0.02` to match the eval variants). Steps ≈ **40k** single-sample SGD
  (tunable from the smoke output).
- **Report (plain text, mobile-readable):** final train/test accuracy,
  `synapse_count`, cumulative edge-steps, wall-time, mean active-edge fraction.
- **Acceptance:** test accuracy clearly above chance — target ≥ ~0.85, minimal
  bar ≥ 0.5 to prove learning — at a synapse count well below fully-connected.

### Network sizing rationale (the explicit "what size" answer)

- Input **64** / output **10** are fixed by the data.
- **Two hidden layers `(64, 32)`.** Digits is near-linearly separable (logistic
  regression ≈ 95%), so two hidden layers are ample and compute-lean. First
  hidden = 64 to match input richness, tapering to 32. (Spirals used 3×16 hidden;
  digits needs more *width* for the 64-d input but not more *depth*.)
- **Density 0.4** (the robust baseline): first-layer fan-in ≈ 0.4 × 64 ≈ 26 —
  healthy for He init — and phasic sleep/startle then adapt connectivity from
  there.
- **Edge budget:** ~6.5k dense → ~2.6k at density 0.4 (~11× the spirals
  baseline's ~230). A single 40k-step run is tens of minutes in pure Python —
  which is exactly why the smoke test is 1 seed and the multi-seed suite is an
  explicit user-run (see Risks), and why MNIST waits on the vectorized backend.

## Stage 3 — dataset registry + harness integration

### New seam: `sprout/datasets.py`

A tiny registry that unifies generated and fixed datasets behind one interface:

```
get_dataset(name, seed, *, n_points, turns, noise)
    -> (X_tr, y_tr, X_te, y_te)
```

- **Generated** (`spirals`, `blobs`): train at `seed`, test at
  `seed + test_seed_offset`, calling the existing generators with identical args
  ⇒ **byte-identical** to today. This protects the pinned baselines and the eval
  disk cache.
- **Fixed** (`digits`): load once, standardize on train stats, seeded stratified
  split (the Stage-1 logic, promoted into the registry).
- This is the single place **MNIST** later plugs in — add one loader branch, the
  harness is untouched.

### Runner integration (`evals/runner.py`)

- `run_one` gets its train/test via `get_dataset` instead of the inline `_gen`
  calls. Spirals/blobs behavior is unchanged (guaranteed by the byte-identity
  test below).
- The `1 − y` label-swap **shift is binary-only**. For digits, `shift_steps`
  stays 0; multiclass concept-shift is out of scope (noted as future work). The
  shift branch is guarded to binary tasks.
- Cache key: dataset + layers + density + seed already capture what matters; the
  digits split fraction is fixed, and `n_points/turns/noise` are inert for the
  fixed dataset.

### Spec / CLI (`evals/spec.py`, `evals/cli.py`)

- `--dataset` choices gain `digits`.
- The digits comparison runs the variants that **don't pin `init_layers`** (so
  they inherit `suite.layers = (64, 64, 32, 10)`): `phasic-startle-k4` (sparse
  baseline) vs `fully-connected` (control), optionally `currency` /
  `static-sparse`. The `eff-*` / `size-w*` probes hard-pin 2-in/2-out and are
  excluded for digits.
- The compute-efficiency story is the existing **FC-vs-sparse** comparison on
  edge-steps / wall-time / active-edge fraction — now on a real 10-class task.

## Testing (TDD)

- **registry/digits:** returns shapes `(?,64)` and labels `0..9`; deterministic
  per seed; stratified split sizes; standardized (≈ zero-mean on train); **no
  leakage** (test uses train μ,σ).
- **regression:** the `spirals`/`blobs` path through `get_dataset` is
  byte-identical (`np.array_equal`) to the previous `_gen` draws — protects the
  eval cache + pinned baselines.
- **learning:** a tiny digits net trained a few hundred steps beats chance
  (acc > 0.1) — fast and deterministic.
- **harness:** `run_one(dataset="digits")` returns a well-formed RunResult; the
  full existing test suite (219 tests) and `validate.py` (7/7) stay green.

## Out of scope (YAGNI)

- Full MNIST (separate future spec; needs the vectorized forward/backward first).
- Multiclass concept-shift / continual-learning on digits.
- Activation-sparsity homeostasis (its own in-flight spec).
- Decision-boundary visualization for >2D inputs.

## Risks

- **Compute:** pure-Python single-sample over ~2.6k edges × 40k steps is slow;
  the smoke test is 1 seed, and the full multi-seed suite is an explicit user-run,
  not part of "build complete".
- **Cache disturbance:** the registry *must* reproduce spiral/blob draws exactly;
  covered by the byte-identity regression test.
