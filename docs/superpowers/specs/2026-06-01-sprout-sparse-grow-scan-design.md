# SPROUT: sparse, activity-aware grow scan (kill the O(N²) ghost enumeration)

- **Date:** 2026-06-01
- **Branch:** sprout-v1
- **Status:** approved (brainstorm), ready to plan + implement
- **Author:** design session with the maintainer

## 1. Problem

The gradient-as-currency growth readout decides which *missing* wire the loss most
wishes existed by scoring "ghost" wires with a RigL-style virtual gradient. Today
[`batch_edge_scores`](../../../sprout/currency.py) does this by enumerating **every
ordered pair of neurons** — an `O(N²)` double loop — for each of `virt_batch=32`
samples, every `t_struct` steps:

```python
for j in range(net.num_neurons):          # all neurons
    for i in range(net.num_neurons):      # × all neurons  → N²
        if layer(i) < layer(j) and (i, j) not live:
            ghost[(i, j)] += |delta_j · activation_i|
```

For the default `w16` net (`N = 52`) that's ~2,700 pairs × 32 samples per grow
cycle. The cost grows with the **square of the neuron count**, so it is the one
piece of training that does not exploit the network's sparsity — and the piece
that blocks scaling to larger nets.

The rest of the per-step path (forward / backward / meters / gated update /
confidence) is already `O(live edges)` — it never touches a dense `N²` matrix —
so it already grows with *edges*, not neuron count. The grow scan is the lone
`O(N²)` offender.

## 2. Goal and non-goals

**Goal.** Make the grow-scan cost scale with **active edges**, not neuron count,
and *demonstrate* it as a published cost-vs-size curve. Growth quality (which
wires get grown) must not regress.

**Non-goals (this project):**
- Vectorising the network to dense/CSR matrices. The per-synapse object model is
  the project's whole point; we keep it.
- Sparsifying the confidence update (its target drifts with global `w̄`/`M̄`, so
  lazy update is hard and the payoff is only constant-factor).
- Touching prune, the confidence rule, or forward/backward.

**Deferred to the architecture doc's roadmap (built later, not now):**
- **#2 lazy meters** — the gradient-meter EMA loops *all* live wires every step,
  but a silent wire's update is pure decay (`M ← β·M`); decay it lazily on next
  touch instead of looping it. Bit-identical, constant-factor.
- **#3 event-driven forward/backward** — propagate only out of neurons that fired
  (active frontier), so silent neurons and their wires cost nothing. Turns
  "∝ edges" into "∝ active edges" for the per-step path. Biggest inner-loop change.

These are recorded as a "Research roadmap: compute efficiency" section in
`docs/v1_implementation.MD`, with the reasoning that they are constant-factor
(already ∝ edges) rather than scaling wins.

## 3. Design

All change is contained in the **growth readout**. One shared candidate-generation
unit encodes "which ghosts do we even look at," and both phases are a change to
*that* unit.

### 3.1 Shared candidate generation (the one source of truth)

New helpers in `sprout/currency.py`:

- `active_ghost_sets(net, grad_b, grow_demand_k=None) -> (active_pre, active_post)`
  - `active_pre  = [i for i in range(N) if neurons[i].activation != 0.0]`
    — note **`!= 0`**, not `> 0`: input neurons hold raw coordinates that can be
    negative, and those ghosts are real (their virtual gradient is nonzero).
  - `active_post = [j for j in grad_b if grad_b[j] != 0.0]` (`grad_b` already
    excludes inputs; a non-firing ReLU hidden neuron has `delta = 0`).
  - If `grow_demand_k` is an int, restrict `active_post` to the **top-k by
    `|grad_b[j]|`** (the loss's loudest complaints) — this is the Phase-2 bound.
- `iter_ghost_candidates(net, active_pre, active_post)` — yields `(i, j)` for
  `j in active_post`, `i in active_pre` with `layer(i) < layer(j)` and
  `(i, j) not in net.synapses`. The single definition of a valid candidate.
- `dense_ghost_count(net) -> int` — closed-form count of valid candidate wires
  (layer-ordered, not live): `Σ_j (neurons in earlier layers − indegree(j))` over
  non-input `j`. `O(N)`, activity-independent, grows ~`N²`. This is the
  measurement baseline (§4).

### 3.2 Phase 1 — exact-sparse (bit-identical, the safe foundation)

Rewrite `batch_edge_scores` to score only `iter_ghost_candidates(...)` (with
`grow_demand_k=None`, i.e. all active posts) instead of the `N²` double loop. Per
batch sample, after `forward`+`backward`:

```python
ap, apo = active_ghost_sets(net, grad_b, grow_demand_k=None)
for (i, j) in iter_ghost_candidates(net, ap, apo):
    ghost[(i, j)] += abs(grad_b[j] * net.neurons[i].activation)
```

`ref` (mean `|gradient|` over live wires) is computed exactly as today. Cost drops
from `O(N²)` to `O(|active_pre| · |active_post|)`.

**Bit-identical claim (precise).** Every pair the new code skips contributed
exactly `0` to a ghost's score before (`activation_i == 0` ⇒ `|delta·0| = 0`;
`delta_j == 0` ⇒ `|0·a| = 0`). So **all nonzero ghost scores, `ref`, and therefore
every grow decision are unchanged.** The one observable difference: today's code
*creates zero-valued ghost keys* for silent-pre pairs (via `ghost[k] += 0.0` on a
`defaultdict`); the new code omits them. Those keys never cleared the grow bar
(`grow_currency` only acts on `score > bar > 0`) and `update_ghost_meter` treats
"absent" and "present-but-0" identically, so grow decisions and the ghost meter
are unaffected. Tests assert equality on the nonzero sub-dict + `ref` + the actual
grown edges, not on the raw dict.

### 3.3 Phase 2 — demand-gated bound (the variant)

New `Config.grow_demand_k: int | None = None`. When set to `k`,
`active_ghost_sets` keeps only the top-k highest-`|delta|` post neurons, capping
work to `k · |active_pre|` per sample. Default `None` ⇒ Phase-1 (exact-sparse)
behavior, so the **baseline is unchanged**.

This *can* change which wires grow (a perpetually-mid-demand neuron may be
under-scored), so it ships as a harness **variant** and is validated against the
baseline for accuracy in the eval (§4), never silently promoted. It preserves
skip-layer growth (we deliberately rejected the layer-local rule, which would
forbid new skip connections).

### 3.4 Threading

`batch_edge_scores` reads `cfg.grow_demand_k`. It is called from
`Trainer._rewire_currency`, which already has `self.cfg` — pass
`cfg.grow_demand_k` through. No change to the per-step loop or history plumbing
(measurement is computed post-hoc; see §4).

## 4. Measurement (the deliverable)

Two **deterministic, descriptive** metrics (direction `neutral` — they are
measurements like `synapse_count`, not goals; the scaling story is told by the
chart and raw numbers, not by ▲/▼ verdicts, which would be misleading across a
size sweep):

- `ghost_dense_cost` — `dense_ghost_count(net)`: candidate ghost wires the scan
  must consider in principle (`O(N²)`). Closed-form on the final topology.
- `ghost_pairs_scored` — mean over the held-out set of
  `len(list(iter_ghost_candidates(net, *active_ghost_sets(net, grad_b, cfg.grow_demand_k))))`
  per sample: candidate wires actually evaluated after activity + demand pruning.

Computed by a new pure function `metrics.ghost_scan_cost(net, X, y, grow_demand_k)`
that **reuses** `sprout.currency.{active_ghost_sets, iter_ghost_candidates,
dense_ghost_count}` — single source of truth, no drift between what training does
and what we measure. Called in `final_snapshot` (which already receives `net`,
`X_test`, `y_test`, `cfg`). Registered in `METRIC_DIRECTIONS` (both `neutral`),
`METRIC_DESCRIPTIONS`, a new `METRIC_FAMILIES["Compute cost"]`, and added to
`evals.publish.KEY_METRICS`.

Each run result also carries `n_neurons` (`net.num_neurons`) for the chart x-axis.

**New chart** `report._plot_scaling(agg, results, path)` → `cost_scaling.png`,
rendered (guarded) by `write_report` only when the cost metrics + `n_neurons` are
present: x = neuron count per variant, two lines — `ghost_dense_cost` (the `N²`
baseline) and `ghost_pairs_scored` (actual) — the widening gap is the result.
`publish_run` auto-discovers the PNG.

### 4.1 Eval runs to publish

1. **Scaling curve (headline).** Variants `size-w4, size-w6, size-w10, size-w16,
   size-w24` (existing; currency at increasing width, `grow_demand_k=None` ⇒
   exact-sparse), baseline `size-w16`. Single regime, spirals, 15k steps, ≥5
   seeds, `--no-cache`. Shows `ghost_dense_cost` (~`N²`) vs `ghost_pairs_scored`
   (∝ active edges) across size, plus the chart. Bonus cross-check: `final_test_acc`
   must match the published `neuron-width-sweep` (Phase 1 is bit-identical).
2. **Demand-gating tradeoff.** Variants `currency` (k=None) vs `currency-bounded`
   (k chosen, e.g. 4) — possibly a second `k` — baseline `currency`, w16, ≥5
   seeds, 15k, `--no-cache`. Verdict must show `final_test_acc ≈` (no accuracy
   loss) while `ghost_pairs_scored` drops further.

Both published via `--publish --run-name`, committed + pushed (pre-authorized),
each with a chat reply: key-metrics table (incl. "What it means"), plain verdict,
git path, honest wins **and** losses.

## 5. Testing (TDD, red first)

- **exact-sparse bit-identity** — tiny net with firing / silent / negative-coord
  input neurons: new `batch_edge_scores` nonzero `ghost` sub-dict + `ref` equal a
  brute-force all-pairs reference.
- **`!= 0` predicate** — a ghost from a negative-coordinate input neuron *is*
  scored (guards the `>0` vs `!=0` trap).
- **demand-gating** — with `grow_demand_k=k`, only ghosts into the top-k `|delta|`
  posts appear, and the per-sample candidate count ≤ `k · |active_pre|`.
- **`dense_ghost_count`** — equals the brute-force count of layer-ordered non-live
  pairs on a hand-checked small net.
- **metrics** — `ghost_pairs_scored ≤ ghost_dense_cost`; both present in
  `final_snapshot`; direction `neutral`; in the "Compute cost" family.
- **end-to-end regression** — currency baseline (`grow_demand_k=None`) produces an
  **identical final synapse set** to the pre-change code on a fixed seed (the
  strongest guard that growth is unchanged).
- **config default** — `Config().grow_demand_k is None`.
- **spec/cli** — the new `currency-bounded` variant exists and is currency +
  `grow_demand_k`.

## 6. File-by-file changes

| File | Change |
|---|---|
| `sprout/currency.py` | Add `active_ghost_sets`, `iter_ghost_candidates`, `dense_ghost_count`; rewrite `batch_edge_scores` to use them + accept `grow_demand_k`. |
| `sprout/train.py` | `Config.grow_demand_k: int \| None = None`; pass it from `_rewire_currency`. |
| `evals/metrics.py` | `ghost_scan_cost`; wire two metrics into `final_snapshot`; register direction/description/family. |
| `evals/runner.py` | Add `n_neurons` to both run results. |
| `evals/report.py` | `_plot_scaling` → `cost_scaling.png`, guarded in `write_report`. |
| `evals/publish.py` | Add the two cost metrics to `KEY_METRICS`. |
| `evals/spec.py` | `currency-bounded` variant (currency + `grow_demand_k`). |
| `docs/v1_implementation.MD` | New "Research roadmap: compute efficiency" section (#2, #3). |
| tests | Per §5, in `test_eval_*` / `test_train` / new `sprout` currency tests. |

## 7. Risks

- **Phase 2 accuracy.** Demand-gating is approximate. Mitigation: it's opt-in
  (`grow_demand_k=None` default), and the eval gates it on `final_test_acc ≈`.
- **Metric/training drift.** Mitigation: the metric reuses the exact same
  candidate helpers training uses — they cannot diverge.
- **Cache staleness.** New metrics aren't in the cache key. Mitigation: run with
  `--no-cache` (standing rule after any metric-schema change).
