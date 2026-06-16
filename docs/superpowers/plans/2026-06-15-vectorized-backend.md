# Vectorized Backend (ArrayNet) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use checkbox (`- [ ]`) tracking.

**Goal:** A pure-NumPy `ArrayNet` that runs SPROUT's per-step wake path vectorized (flat edge-list + scatter/segment-sums), faithful to the object `Network`, with a round-trip to the object model for rare phasic prune/grow bursts.

**Architecture:** Edges as parallel arrays + per-layer index groups; forward/backward as scatter-adds in topological order; per-step meters/confidence/gated-update as array ops. Structural bursts `sync_into(net)` → reuse `currency.prune/grow` → `from_network` rebuild.

**Tech Stack:** Python, NumPy, existing `sprout` modules. Tests via `.venv/bin/python -m pytest`.

**Files:** Create `sprout/fast.py`, `tests/test_fast.py`, `bench_fast.py`. Object path untouched.

---

### Task 1: `ArrayNet` representation + round-trip converters

- [ ] **Step 1 (test, `tests/test_fast.py`):** build a graph, `ArrayNet.from_network`, mutate arrays, `sync_into`, assert the object net got the values; assert edge order matches `list(net.synapses)`.

```python
import numpy as np
from sprout.network import build_graph, init_weights
from sprout.fast import ArrayNet

def _net(seed=0, layers=(4, 6, 3), density=0.6):
    net = build_graph(list(layers), density=density, seed=seed)
    init_weights(net, seed=seed)
    return net

def test_from_network_and_sync_roundtrip():
    net = _net()
    an = ArrayNet.from_network(net)
    assert an.keys == list(net.synapses.keys())
    assert an.E == len(net.synapses) and an.N == net.num_neurons
    an.weight += 1.5
    an.confidence[:] = 0.25
    an.age += 3
    an.sync_into(net)
    for r, k in enumerate(an.keys):
        s = net.synapses[k]
        assert abs(s.weight - an.weight[r]) < 1e-12
        assert abs(s.confidence - 0.25) < 1e-12 and s.age == an.age[r]
```

- [ ] **Step 2:** run → fail (no module). `.venv/bin/python -m pytest tests/test_fast.py -q`
- [ ] **Step 3:** implement `ArrayNet.__init__/_build/from_network/sync_into` per the spec data layout (neuron arrays bias/layer/firing_rate, edge arrays in `net.synapses` order, `in_edges[L]`=rows with post in layer L, `out_edges[L]`=rows with pre in layer L, `keys` list, buffers activation/delta).
- [ ] **Step 4:** run → pass.
- [ ] **Step 5:** commit `feat: ArrayNet representation + round-trip converters`.

### Task 2: vectorized `forward` parity

- [ ] **Step 1 (test):** for a net, `net.forward(x)` (object) vs `ArrayNet.from_network(net).forward(x)`; assert probs allclose `1e-9` AND hidden activations allclose (compare `an.activation` to object `neuron.activation`). Use a few random x (incl. negative coords).

```python
def test_forward_matches_object():
    net = _net(layers=(4, 8, 5, 3), density=0.5)
    an = ArrayNet.from_network(net)
    rng = np.random.default_rng(1)
    for _ in range(5):
        x = rng.normal(size=4)
        p = net.forward(x); pa = an.forward(x)
        assert np.allclose(p, pa, atol=1e-9)
        obj_a = np.array([n.activation for n in net.neurons])
        assert np.allclose(obj_a, an.activation, atol=1e-9)
```

- [ ] **Step 2:** run → fail. **Step 3:** implement `forward` (global z buffer via `np.bincount(post[e], contrib, minlength=N)`; ReLU; optional WTA top-k matching `network.forward`; output softmax via a local `_softmax`). **Step 4:** pass. **Step 5:** commit `feat: ArrayNet.forward (scatter-add, parity with object)`.

### Task 3: vectorized `backward` parity

- [ ] **Step 1 (test):** after a shared forward, compare `grad_w` (object dict → array in `an.keys` order) and `grad_b`/delta and loss, allclose `1e-9`.

```python
def test_backward_matches_object():
    net = _net(layers=(4, 8, 5, 3), density=0.5)
    an = ArrayNet.from_network(net)
    x = np.random.default_rng(2).normal(size=4); y = 1
    net.forward(x); loss_o, gw_o, gb_o = net.backward(y)
    an.forward(x); loss_a, gw_a = an.backward(y)
    assert abs(loss_o - loss_a) < 1e-9
    gw_o_arr = np.array([gw_o[k] for k in an.keys])
    assert np.allclose(gw_o_arr, gw_a, atol=1e-9)
    gb_a = an.delta
    for nid, g in gb_o.items():
        assert abs(g - gb_a[nid]) < 1e-9
```

- [ ] **Step 2:** fail. **Step 3:** implement `backward` (output delta=probs-onehot; reverse layers, `acc=bincount(pre[e], weight[e]*delta[post[e]])`, `delta[ids]=acc[ids]*(activation[ids]>0)`; return `loss, grad_w=delta[post]*activation[pre]`; expose grad_b via `self.delta`). **Step 4:** pass. **Step 5:** commit `feat: ArrayNet.backward (parity with object)`.

### Task 4: vectorized settledness + per-step `step` parity (the key test)

- [ ] **Step 1 (test A — settledness):** vectorized `_settledness_vec(d, mode, k)` matches scalar `currency.settledness` elementwise across a grid of d in [0,3] for modes hard/sigmoid/exp/rational.
- [ ] **Step 1 (test B — multi-step wake parity):** build identical net+config (`phasic-startle-k4`-style: enable_confidence/prune/grow True, twod, sigmoid, beta_g, eta 0.02) and the SAME data; drive the **object** wake path (forward, update_firing_rates, backward, update_gradient_meters, update_confidence_2d, apply_gated_update, increment ages) for K=200 steps on a fixed sample stream, and `ArrayNet.step` on the same stream; assert final `weight`, `confidence`, `grad_mag` allclose `atol=1e-7` and predictions identical on a probe set. (No structural change in this test.)

```python
def test_wake_step_parity():
    from sprout.train import Config
    from sprout import learning, currency
    cfg = Config(eta_base=0.02, enable_confidence=True, enable_prune=True,
                 enable_grow=True, gamma_dec=0.001, t_struct=200,
                 phasic_structure=True, startle=True, grow_demand_k=4)
    net = _net(layers=(4, 8, 6, 3), density=0.5)
    an = ArrayNet.from_network(net)
    rng = np.random.default_rng(7); X = rng.normal(size=(64, 4)); Y = rng.integers(0, 3, 64)
    order = rng.integers(0, 64, size=200)
    for s in order:                       # object wake path
        net.forward(X[s]); learning.update_firing_rates(net, cfg.beta)
        _, gw, gb = net.backward(int(Y[s]))
        currency.update_gradient_meters(net, gw, cfg.beta_g, step_idx=_step(net))
        currency.update_confidence_2d(net, cfg.conf_gain, cfg.conf_alpha, cfg.c_max,
                                      cfg.settled_mode, cfg.conf_k)
        learning.apply_gated_update(net, gw, gb, cfg.eta_base)
        for syn in net.synapses.values(): syn.age += 1
    # ... see implementation note; simpler: drive object via a Trainer-less helper
```

  *Implementation note:* to avoid step-index bookkeeping drift, the test drives **both** sides through a tiny shared `_wake_step(...)` helper defined in the test for the object side (mirroring `Trainer._step_currency`'s order with `step_idx`), and `ArrayNet.step(x, y, cfg, step_idx)` for the array side, with `step_idx` incremented identically.

- [ ] **Step 2:** fail. **Step 3:** implement `_settledness_vec` and `ArrayNet.step(x, y, cfg, step_idx)` exactly in `Trainer._step_currency` order: forward → firing_rate EMA (all neurons) → backward → meters (non-lazy: `grad_mag=βg·grad_mag+(1-βg)|grad_w|`, `grad_signed` likewise, `grad_last_step=step_idx+1`) → confidence (2D or tugofwar) → gated update (`weight-=eta/(1+confidence)·grad_w`; `bias[noninput]-=eta·delta`) → `age+=1`; return loss. **Step 4:** pass (tune tol to `1e-7` if needed; if drift, also assert predictions identical). **Step 5:** commit `feat: ArrayNet.step wake-path + vectorized settledness (parity vs object)`.

### Task 5: structural round-trip + `train_array` runner + end-to-end

- [ ] **Step 1 (test):** `train_array(cfg, net, X, y, seed, steps)` on spirals (small net) returns a trained net with test accuracy > 0.7 and `synapse_count_end < synapse_count_start` (prune fired); and is deterministic for a fixed seed.

```python
def test_train_array_learns_and_sparsifies():
    from sprout.train import Config, accuracy
    from sprout.data import generate_spirals
    from sprout.fast import train_array
    cfg = Config(eta_base=0.02, enable_confidence=True, enable_prune=True,
                 enable_grow=True, gamma_dec=0.001, t_struct=200,
                 phasic_structure=True, startle=True, grow_demand_k=4,
                 sleep_warmup=500, sleep_patience=300)
    X, y = generate_spirals(n=400, seed=0)
    net = build_graph([2, 12, 12, 2], density=0.5, seed=0); init_weights(net, seed=0)
    start = len(net.synapses)
    net = train_array(cfg, net, X, y, seed=0, steps=6000)
    assert accuracy(net, X, y) > 0.7
    assert len(net.synapses) < start
```

- [ ] **Step 2:** fail. **Step 3:** implement `train_array`: build ArrayNet; per step `an.step`; feed loss to `sleep.SettlednessDetector`; on `settled` (phasic) → `an.sync_into(net)` → reuse the rewire (call `currency.prune_currency` + `currency.grow_currency` exactly as `_rewire_phasic`, with `batch_edge_scores` over a virt batch) → `an = ArrayNet.from_network(net)` → `detector.reset()`. Sample stream via `np.random.default_rng(seed)` matching the Trainer. After the loop, `an.sync_into(net)` and return net. **Step 4:** pass. **Step 5:** commit `feat: train_array runner (vectorized wake + object-reuse phasic bursts)`.

### Task 6: benchmark + report

- [ ] **Step 1:** write `bench_fast.py`: for edge counts via widths (e.g. `(2,W,W,2)`/`(196,W,10)` at densities giving ~300, ~1k, ~3k, ~10k, ~30k edges), time 300 object wake steps vs 300 `ArrayNet.step` (same net/data), print edges, object ms/step, array ms/step, speedup.
- [ ] **Step 2:** run `.venv/bin/python bench_fast.py`; record the table.
- [ ] **Step 3:** full suite `.venv/bin/python -m pytest -q` (object path untouched → green) + `.venv/bin/python validate.py` (7/7). Commit `bench: vectorized backend speedup measurement` + report numbers.

---

## Self-Review
- **Spec coverage:** representation/converters → T1; forward → T2; backward → T3; meters+confidence+gated+firing+age+settledness → T4; structural round-trip + runner + end-to-end → T5; benchmark + regression → T6. ✅
- **Placeholders:** none — each task has concrete code/commands (T4's object-side driver is a small shared helper, noted explicitly).
- **Type consistency:** `ArrayNet.from_network`, `.forward`, `.backward`→`(loss, grad_w)` with grad_b via `.delta`, `.step(x,y,cfg,step_idx)`, `.sync_into(net)`, `train_array(cfg,net,X,y,seed,steps)` consistent across tasks.
- **Out of scope (per spec):** minibatch, SciPy/GPU, lazy-meter vectorization, harness `--backend` wiring (phase 2).
