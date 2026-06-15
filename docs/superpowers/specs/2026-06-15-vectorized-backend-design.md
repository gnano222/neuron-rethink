# Vectorized backend for SPROUT (`ArrayNet`)

**Date:** 2026-06-15 · **Branch:** sprout-v1 · **Status:** approved (Approach 1, autonomous build)

## Goal & constraints

Replace the per-synapse **Python scalar loops** in the per-step training path with
**vectorized NumPy array ops**, so the same sparse self-rewiring network runs
~10–50× faster and **bigger edge budgets become feasible** (the "increase the
budget" lever, currently capped by Python speed). Decisions locked in
brainstorming:

- **Faithful parallel backend.** A new array-based engine; the object `Network` +
  `currency.py` stay the readable **reference** (and the `validate.py` guardrail).
  The array path must match the reference within float tolerance so existing
  findings transfer and A/B stays valid. Opt-in, like every SPROUT promotion.
- **Single-sample.** One sample per step — identical online dynamics. The speedup
  comes from vectorizing across **edges**, not from minibatching (minibatching
  changes the learning regime and is explicitly out of scope; separate future spec).
- **Approach 1: flat edge-list + scatter/segment-sums (pure NumPy, no new deps).**

## Why a flat edge list (not per-layer matrices)

Growth creates **skip connections** (`iter_ghost_candidates` allows any
`layer(i) < layer(j)`, not just adjacent), so the topology is a general layered
DAG. A flat edge list with `np.bincount`/`np.add.at` scatter handles that
naturally and keeps **per-synapse semantics** (each edge is an array row; every
attribute — weight, confidence, age, meters — is a parallel array). Structural
edits are cheap (mask to prune, append to grow) and rare (phasic). SciPy CSR is a
possible future swap-in behind the same interface; dense masked matrices are
rejected (they restore `O(N²)` and discard the sparsity that is the point).

## Architecture

New module `sprout/fast.py` with one class, `ArrayNet`, built from a `Network`.

### Data (parallel arrays)
- Neurons (`N` = num_neurons): `bias[N] float`, `layer[N] int`, plus per-step
  buffers `activation[N]`, `delta[N]`. `input_ids`, and `layer_neurons[L]` = int
  array of neuron ids per layer.
- Edges (`E` = num_synapses, a fixed row order): `pre[E] int`, `post[E] int`,
  `weight[E]`, `confidence[E]`, `age[E] int`, `grad_mag[E]`, `grad_signed[E]`,
  `grad_last_step[E] int`.
- Precomputed index groups (rebuilt only on structural change):
  - `in_edges[L]` = edge rows whose **post** ∈ layer L (for forward).
  - `out_edges[L]` = edge rows whose **pre** ∈ layer L (for backward).
  - `edge_key_to_row: dict[(pre,post)->row]` for sync/lookup.

### Conversion (the round-trip)
- `ArrayNet.from_network(net)` — read object → arrays (stable edge order =
  `net.synapses` insertion order, so grad/attribute order matches the reference).
- `ArrayNet.sync_into(net)` — write array state (weight, confidence, age,
  grad_mag/signed/last_step, bias, firing_rate) back into the object `Network`.
  Used only at structural bursts.

## The vectorized per-step path (must mirror `Trainer._step_currency` order)

1. **forward(x)** — `activation[input_ids] = x`; for each hidden layer L in order:
   `z = bias[layer_neurons[L]]` then scatter-add `weight[e]*activation[pre[e]]`
   over `e = in_edges[L]` (via `np.add.at` on a z buffer), `activation = relu(z)`;
   optional layer-local top-k WTA (zero all but top-k positives, matching
   `network.forward`). Output layer: scatter-add then softmax. Returns probs.
   *(A neuron's incoming edges may come from any earlier layer — all already
   computed in topological order, so one scatter per layer is correct.)*
2. **firing rates** — `firing_rate = (1-β)·firing_rate + β·activation` over **all**
   neurons (matches `update_firing_rates`, incl. inputs/outputs — it feeds
   dead-unit detection).
3. **backward(y)** — `loss = -log(probs[y]+1e-12)`; `delta=0`; `delta[out] =
   probs - onehot`; for L from last-1 down to 1: `acc = bincount(pre[e],
   weight[e]·delta[post[e]])` over `e = out_edges[L]`, then `delta[layer L] = acc ·
   (activation[layer L] > 0)` (ReLU′ keyed on **activation**, so WTA-suppressed
   units get 0, matching the reference). Edge gradient `grad_w = delta[post] ·
   activation[pre]` (one array). `grad_b = delta` (non-input).
4. **meters** — `grad_mag = β_g·grad_mag + (1-β_g)·|grad_w|`; `grad_signed`
   likewise (vectorized `update_gradient_meters`; lazy variant deferred — non-lazy
   first since `lazy_meters=False` is the default).
5. **confidence (2D)** — `wbar=mean|weight|`, `mbar=mean grad_mag`; `load=|w|/(wbar
   +eps)`; `imp=max(load-1,0)`; `settled=settledness(grad_mag/(mbar+eps))`;
   `target=min(gain·imp·settled,c_max)`; `confidence=(1-α)·confidence+α·target`.
   (Reuse the scalar `settledness` from currency.py, vectorized over the array.)
6. **gated update** — `weight -= (eta/(1+confidence))·grad_w`; `bias[j] -=
   eta·grad_b[j]` for non-input j.
7. **ages** — `age += 1`.
8. **settledness detector** — reuse `sprout.sleep.SettlednessDetector` unchanged
   (it only consumes the scalar loss).

## Structural bursts (rare — reuse the reference, don't reimplement)

When the detector says *settled* (the phasic plateau): `sync_into(net)` → run the
**existing** object-side rewire (`prune_currency` / `grow_currency`, i.e. the
already-tested `_rewire_phasic` logic) → `ArrayNet.from_network(net)` to rebuild
the arrays + index groups. The conversion is paid only at bursts (far apart), so
it is negligible amortized, and the structural behavior is **exactly** the
reference's (no re-validation of prune/grow needed).

## Integration

- `sprout/fast.py` is self-contained: `ArrayNet` exposes `forward`, `backward`,
  and a `step(...)`-style per-step update, plus the converters.
- A thin runner `ArrayNet.train(cfg, X, y, seed, steps)` (mirrors the object
  Trainer's loop: vectorized wake steps + the sync/rewire/rebuild burst on settle)
  so a full single-task run works end-to-end on the fast path.
- Trainer/eval-harness wiring behind a `backend` flag is a **follow-up** (the
  engine + a standalone runner is the deliverable here; harness integration is
  scoped in the plan as phase 2).

## Testing (the correctness contract)

- **Unit:** `from_network`/`sync_into` round-trip preserves all arrays; forward
  reproduces `network.forward` activations within `1e-9`; backward reproduces
  `grad_w`/`grad_b` within `1e-9` on a fixed small net.
- **Parity (the key test):** from one seed/data/init, run K wake steps (no
  structure) with the object `Trainer` path **and** `ArrayNet`; assert final
  weights/confidence/grad_mag match within tolerance (`atol~1e-8`) and predictions
  are identical. Float summation order differs (scatter vs sequential), so this is
  *close*, not bit-identical — the test asserts a tight tolerance, not equality.
- **End-to-end:** a full `ArrayNet.train` run (wake + phasic bursts via round-trip)
  on spirals learns (accuracy ≫ chance) and ends sparser (prune fired).
- **Regression:** full existing suite stays green (object path untouched).

## Benchmark (the payoff, reported at the end)

Time the per-step path, object vs `ArrayNet`, across edge counts (~300 → ~50k) on
one net; report the speedup curve + ms/step. Success = a clear, growing speedup
that makes ~10–50× larger budgets practical.

## Out of scope (YAGNI)

- Minibatching; SciPy/GPU backends; lazy-meter vectorization (non-lazy first);
  full Trainer/eval `--backend` wiring (phase 2); convolution/weight-sharing
  (a different project entirely).

## Risks

- **Float drift** between scatter and sequential sums → parity uses tolerances,
  not equality; if drift compounds over many steps, tighten by matching reduction
  order where feasible and assert on predictions/accuracy as the real bar.
- **Index-group rebuild correctness** on prune/grow → covered by the end-to-end
  test (a run that prunes and grows must still match a reference run's trajectory
  qualitatively).
