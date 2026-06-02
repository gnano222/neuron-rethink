# Sparse activity-aware grow scan — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the grow readout's `O(N²)` ghost-wire enumeration with an activity-sparse scan whose cost scales with active edges, and publish the cost-vs-size curve — without changing which wires grow at the default.

**Architecture:** One shared candidate-generation unit in `sprout/currency.py` (`active_ghost_sets`, `iter_ghost_candidates`, `dense_ghost_count`) is the single source of truth for "which ghosts do we look at." `batch_edge_scores` (Phase 1, bit-identical) and a new compute-cost metric both consume it. A `Config.grow_demand_k` knob adds the Phase-2 demand-gated bound as an opt-in variant. Measurement is post-hoc (a pure metric on the final net), so the training loop is untouched beyond passing the knob through.

**Tech Stack:** Python 3, numpy, matplotlib (Agg), pytest. Run python via `.venv/bin/python` and tests via `.venv/bin/python -m pytest`.

---

### Task 1: Shared candidate helpers in `sprout/currency.py`

**Files:**
- Modify: `sprout/currency.py` (add three functions near the Readout C section, before `batch_edge_scores` at line ~221)
- Test: `tests/test_currency.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_currency.py` (and add `active_ghost_sets, iter_ghost_candidates, dense_ghost_count` to the `from sprout.currency import (...)` block at the top):

```python
def test_active_ghost_sets_uses_nonzero_not_positive_for_pre():
    # input coords can be negative; a negative-activation pre is still "active"
    net = Network([2, 2])                      # inputs 0,1 ; outputs 2,3
    net.neurons[0].activation = -0.7           # negative -> still active
    net.neurons[1].activation = 0.0            # exactly zero -> inactive
    grad_b = {2: 0.5, 3: 0.0}                   # only neuron 2 has gradient
    ap, apo = active_ghost_sets(net, grad_b)
    assert 0 in ap and 1 not in ap
    assert apo == [2]                           # 3 dropped (delta 0)


def test_active_ghost_sets_topk_keeps_loudest_posts():
    net = Network([2, 3])                       # inputs 0,1 ; outputs 2,3,4
    for nid in (0, 1):
        net.neurons[nid].activation = 1.0
    grad_b = {2: 0.1, 3: 0.9, 4: 0.5}
    _, apo = active_ghost_sets(net, grad_b, grow_demand_k=2)
    assert set(apo) == {3, 4}                   # top-2 by |delta|


def test_iter_ghost_candidates_respects_layer_order_and_liveness():
    net = Network([2, 1, 1])                    # 0,1 | 2 | 3
    net.add_synapse(2, 3)                       # live
    cand = set(iter_ghost_candidates(net, active_pre=[0, 1, 2], active_post=[2, 3]))
    # into 2 (layer1): 0,1 (layer0) -> (0,2),(1,2). into 3 (layer2): 0,1,2 minus live (2,3)
    assert cand == {(0, 2), (1, 2), (0, 3), (1, 3)}


def test_dense_ghost_count_matches_bruteforce():
    net = build_graph([2, 3, 2], density=0.5, seed=2)
    brute = sum(1 for j in range(net.num_neurons)
                for i in range(net.num_neurons)
                if net.neurons[i].layer < net.neurons[j].layer
                and (i, j) not in net.synapses)
    assert dense_ghost_count(net) == brute
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_currency.py -k "active_ghost_sets or iter_ghost_candidates or dense_ghost_count" -q`
Expected: FAIL — `ImportError: cannot import name 'active_ghost_sets'`.

- [ ] **Step 3: Implement the helpers**

In `sprout/currency.py`, add immediately above `def batch_edge_scores`:

```python
# -- Readout C: shared candidate generation (one source of truth) ------------

def active_ghost_sets(net, grad_b, grow_demand_k=None):
    """Active pre/post neuron-id lists for ghost scoring.

    ``active_pre``  = neurons with **nonzero** activation. Input neurons hold raw
                      coordinates that can be negative, so the predicate is
                      ``!= 0`` (not ``> 0``): those ghosts have a real gradient.
    ``active_post`` = non-input neurons whose backprop delta is nonzero (a silent
                      ReLU unit has ``delta = 0``). If ``grow_demand_k`` is set,
                      keep only the top-k by ``|delta|`` — the demand bound.
    """
    active_pre = [i for i in range(net.num_neurons)
                  if net.neurons[i].activation != 0.0]
    posts = [(j, abs(dj)) for j, dj in grad_b.items() if dj != 0.0]
    if grow_demand_k is not None and len(posts) > grow_demand_k:
        posts.sort(key=lambda jd: jd[1], reverse=True)
        posts = posts[:grow_demand_k]
    return active_pre, [j for j, _ in posts]


def iter_ghost_candidates(net, active_pre, active_post):
    """Yield valid candidate ghost wires: layer-ordered ``(i, j)`` not already live."""
    neurons, syn = net.neurons, net.synapses
    for j in active_post:
        lj = neurons[j].layer
        for i in active_pre:
            if neurons[i].layer < lj and (i, j) not in syn:
                yield (i, j)


def dense_ghost_count(net):
    """Closed-form count of valid candidate wires if *every* neuron were active.

        Sum over non-input j of (neurons in earlier layers) - indegree(j).

    O(N) and activity-independent; grows ~N^2. The measurement baseline the
    activity-sparse scan is compared against.
    """
    sizes = [len(L) for L in net.layers]
    before = 0
    total = 0
    for L in range(1, len(net.layers)):
        before += sizes[L - 1]                 # neurons strictly before layer L
        for j in net.layers[L]:
            total += before - len(net.incoming[j])
    return total
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_currency.py -k "active_ghost_sets or iter_ghost_candidates or dense_ghost_count" -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add sprout/currency.py tests/test_currency.py
git commit -m "currency: shared ghost-candidate helpers (active sets, iterator, dense count)"
```

---

### Task 2: Rewrite `batch_edge_scores` to exact-sparse + `grow_demand_k`

**Files:**
- Modify: `sprout/currency.py` (`batch_edge_scores`, lines ~221-261)
- Test: `tests/test_currency.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_currency.py`:

```python
def _brute_edge_scores(net, X, y):
    """The original O(N^2) enumeration, kept here as the bit-identity oracle."""
    from collections import defaultdict
    ghost = defaultdict(float)
    live_sum = live_n = 0.0
    n = len(X)
    for xi, yi in zip(X, y):
        net.forward(xi)
        _, gw, gb = net.backward(int(yi))
        for g in gw.values():
            live_sum += abs(g); live_n += 1
        for j in range(net.num_neurons):
            lj = net.neurons[j].layer
            if lj == 0:
                continue
            dj = gb.get(j, 0.0)
            if dj == 0.0:
                continue
            for i in range(net.num_neurons):
                if net.neurons[i].layer >= lj or (i, j) in net.synapses:
                    continue
                ghost[(i, j)] += abs(dj * net.neurons[i].activation)
    ghost = {k: v / n for k, v in ghost.items()}
    return ghost, (live_sum / live_n if live_n else 0.0)


def test_batch_edge_scores_bit_identical_to_bruteforce():
    net = build_graph([2, 4, 4, 2], density=0.5, seed=3)
    init_weights(net, seed=3)
    X, y = generate_blobs(n=24, seed=1)
    g_new, ref_new = batch_edge_scores(net, X, y)
    g_old, ref_old = _brute_edge_scores(net, X, y)
    nz_old = {k: v for k, v in g_old.items() if v != 0.0}   # drop spurious 0 keys
    assert ref_new == pytest.approx(ref_old, rel=1e-12)
    assert set(g_new) == set(nz_old)
    for k in g_new:
        assert g_new[k] == pytest.approx(nz_old[k], rel=1e-12)


def test_batch_edge_scores_scores_ghost_from_negative_input():
    # a negative input coordinate must still produce a (nonzero) ghost score
    net = build_graph([2, 2, 2], density=0.5, seed=4)
    init_weights(net, seed=4)
    X = np.array([[-0.9, -0.8], [-0.7, 0.6]])
    y = np.array([0, 1])
    g, _ = batch_edge_scores(net, X, y)
    assert any(net.neurons[i].layer == 0 for (i, j) in g)   # an input is a live pre


def test_batch_edge_scores_demand_k_caps_candidates():
    net = build_graph([2, 6, 6, 2], density=0.5, seed=5)
    init_weights(net, seed=5)
    X, y = generate_blobs(n=20, seed=2)
    g_full, _ = batch_edge_scores(net, X, y)
    g_k1, _ = batch_edge_scores(net, X, y, grow_demand_k=1)
    assert len(g_k1) <= len(g_full)                          # bound only removes
    assert set(g_k1) <= set(g_full)                          # subset of the full set
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_currency.py -k "batch_edge_scores" -q`
Expected: FAIL — `test_batch_edge_scores_demand_k_caps_candidates` errors (`batch_edge_scores() got an unexpected keyword argument 'grow_demand_k'`).

- [ ] **Step 3: Rewrite `batch_edge_scores`**

Replace the body of `batch_edge_scores` in `sprout/currency.py` (keep the docstring, update the signature + return shape note):

```python
def batch_edge_scores(net, X, y, grow_demand_k=None):
    """Score candidate (missing) wires by their virtual gradient over a batch.

        g_ij^virt = delta_j * a_i   for active (i,j), layer(i) < layer(j)
        score_ij  = mean over the batch of |g_ij^virt|

    Only activity-relevant candidates are visited: a ghost is nonzero only when
    its pre fired (activation != 0) AND its post has gradient (delta != 0), so the
    scan runs over ``active_pre x active_post`` (see ``active_ghost_sets`` /
    ``iter_ghost_candidates``) rather than all N^2 pairs. With ``grow_demand_k``
    set, only the top-k highest-|delta| posts are considered (the demand bound).

    Returns ``(ghost_scores, ref)`` where ``ref`` is the mean |gradient| over live
    wires (the growth-bar calibration scale).
    """
    ghost = defaultdict(float)
    live_sum, live_n = 0.0, 0
    n = len(X)
    for xi, yi in zip(X, y):
        net.forward(xi)
        _, grad_w, grad_b = net.backward(int(yi))
        for g in grad_w.values():
            live_sum += abs(g)
            live_n += 1
        active_pre, active_post = active_ghost_sets(net, grad_b, grow_demand_k)
        for (i, j) in iter_ghost_candidates(net, active_pre, active_post):
            ghost[(i, j)] += abs(grad_b[j] * net.neurons[i].activation)

    ghost = {k: v / n for k, v in ghost.items()} if n else {}
    ref = (live_sum / live_n) if live_n else 0.0
    return ghost, ref
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_currency.py -q`
Expected: PASS (all currency tests, including the pre-existing `test_batch_scores_are_zero_for_ghosts_into_dead_neuron`).

- [ ] **Step 5: Commit**

```bash
git add sprout/currency.py tests/test_currency.py
git commit -m "currency: exact-sparse batch_edge_scores (bit-identical) + grow_demand_k bound"
```

---

### Task 3: `Config.grow_demand_k` and thread it through the trainer

**Files:**
- Modify: `sprout/train.py` (`Config`, ~line 80; `_rewire_currency`, ~line 292)
- Test: `tests/test_train.py`, `tests/test_currency.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_train.py`:

```python
def test_default_config_has_no_grow_demand_k():
    assert Config().grow_demand_k is None
```

Add to `tests/test_currency.py`:

```python
def test_trainer_passes_grow_demand_k_to_growth():
    net = build_graph([2, 6, 6, 2], density=0.6, seed=6)
    init_weights(net, seed=6)
    X, y = generate_blobs(n=60, seed=3)
    cfg = Config(grad_currency=True, enable_confidence=True, enable_prune=True,
                 enable_grow=True, t_struct=10, grow_demand_k=1)
    tr = Trainer(cfg, net, X, y, seed=0)
    for _ in range(50):
        tr.step()                                # must not raise; growth bounded
    assert isinstance(len(net.synapses), int)    # trained, net intact
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_train.py::test_default_config_has_no_grow_demand_k tests/test_currency.py::test_trainer_passes_grow_demand_k_to_growth -q`
Expected: FAIL — `Config` has no `grow_demand_k` (AttributeError / TypeError on the kwarg).

- [ ] **Step 3: Add the field and pass it through**

In `sprout/train.py` `Config`, after the `grow_bar_frac` / `virt_batch` block (right before the `ghost_meter` field, ~line 70):

```python
    # Phase-2 demand bound for the grow scan: when an int k, score ghosts only
    # into the top-k highest-|delta| post neurons (bounds work to k * active_pre).
    # None => exact-sparse scan over all active posts (the bit-identical default).
    grow_demand_k: int | None = None
```

In `sprout/train.py` `_rewire_currency`, change the `batch_edge_scores` call:

```python
            ghost, ref = batch_edge_scores(net, self.X[idx], self.y[idx],
                                           grow_demand_k=cfg.grow_demand_k)
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_train.py::test_default_config_has_no_grow_demand_k tests/test_currency.py::test_trainer_passes_grow_demand_k_to_growth -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sprout/train.py tests/test_train.py tests/test_currency.py
git commit -m "train: Config.grow_demand_k threaded into the currency grow scan"
```

---

### Task 4: `ghost_scan_cost` metric + wire into `final_snapshot`

**Files:**
- Modify: `evals/metrics.py` (`ghost_scan_cost` near the synapse-quality block; `final_snapshot`; `METRIC_DIRECTIONS`, `METRIC_DESCRIPTIONS`, `METRIC_FAMILIES`)
- Test: `tests/test_eval_metrics.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_eval_metrics.py` (it already imports `from sprout.network import Network` and `from evals import metrics`; also add `from sprout.network import build_graph, init_weights` and `from sprout.data import generate_blobs`):

```python
def test_ghost_scan_cost_scored_le_dense_and_dense_matches_bruteforce():
    net = build_graph([2, 4, 4, 2], density=0.5, seed=7)
    init_weights(net, seed=7)
    X, y = generate_blobs(n=16, seed=4)
    out = metrics.ghost_scan_cost(net, X, y)
    brute_dense = sum(1 for j in range(net.num_neurons)
                      for i in range(net.num_neurons)
                      if net.neurons[i].layer < net.neurons[j].layer
                      and (i, j) not in net.synapses)
    assert out["ghost_dense_cost"] == float(brute_dense)
    assert 0.0 <= out["ghost_pairs_scored"] <= out["ghost_dense_cost"]


def test_ghost_scan_cost_empty_set_is_zero_scored():
    net = build_graph([2, 3, 2], density=0.5, seed=8)
    out = metrics.ghost_scan_cost(net, [], [])
    assert out["ghost_pairs_scored"] == 0.0
    assert out["ghost_dense_cost"] > 0.0


def test_cost_metrics_are_registered_neutral_and_in_compute_family():
    for k in ("ghost_dense_cost", "ghost_pairs_scored"):
        assert metrics.METRIC_DIRECTIONS[k] == "neutral"
        assert k in metrics.METRIC_DESCRIPTIONS
    assert metrics.METRIC_FAMILIES["Compute cost"] == (
        "ghost_dense_cost", "ghost_pairs_scored")
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_eval_metrics.py -k "ghost_scan_cost or cost_metrics" -q`
Expected: FAIL — `metrics.ghost_scan_cost` does not exist / `KeyError: 'Compute cost'`.

- [ ] **Step 3: Implement the metric + registration**

In `evals/metrics.py`, add after `neuron_activation_stats` (the synapse-quality / capacity block):

```python
def ghost_scan_cost(net, X, y, grow_demand_k=None) -> dict:
    """Compute cost of one grow scan on this net: candidate ghost wires.

    ``ghost_dense_cost`` is the activity-independent count of valid candidates
    (~N^2); ``ghost_pairs_scored`` is the mean per-sample count actually evaluated
    after activity + demand pruning. Reuses the *same* candidate helpers the
    trainer's grow scan uses, so measurement cannot drift from behaviour.
    """
    from sprout.currency import (active_ghost_sets, iter_ghost_candidates,
                                  dense_ghost_count)
    dense = float(dense_ghost_count(net))
    if len(X) == 0:
        return {"ghost_pairs_scored": 0.0, "ghost_dense_cost": dense}
    total = 0
    for xi, yi in zip(X, y):
        net.forward(xi)
        _, _, grad_b = net.backward(int(yi))
        ap, apo = active_ghost_sets(net, grad_b, grow_demand_k)
        total += sum(1 for _ in iter_ghost_candidates(net, ap, apo))
    return {"ghost_pairs_scored": total / len(X), "ghost_dense_cost": dense}
```

In `final_snapshot`, after `nact = neuron_activation_stats(net, X_test)`:

```python
    scan = ghost_scan_cost(net, X_test, y_test,
                           getattr(cfg, "grow_demand_k", None))
```

and in the `final = {...}` dict, after the `"mean_neuron_activation": ...` line:

```python
        "ghost_dense_cost": scan["ghost_dense_cost"],
        "ghost_pairs_scored": scan["ghost_pairs_scored"],
```

In `METRIC_DIRECTIONS`, add (after the `meter_fidelity` sanity entry or near it):

```python
    # E. compute cost (descriptive: the scaling story is the chart, not verdicts)
    "ghost_dense_cost": "neutral",
    "ghost_pairs_scored": "neutral",
```

In `METRIC_DESCRIPTIONS`, add:

```python
    "ghost_dense_cost": "candidate ghost wires the grow-scan must consider (~N²)",
    "ghost_pairs_scored": "candidate wires actually scored after activity+demand pruning",
```

In `METRIC_FAMILIES`, add a new entry before `"Signal sanity"`:

```python
    "Compute cost": ("ghost_dense_cost", "ghost_pairs_scored"),
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_eval_metrics.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add evals/metrics.py tests/test_eval_metrics.py
git commit -m "metrics: ghost_scan_cost (dense vs scored) + Compute cost family"
```

---

### Task 5: Add `n_neurons` to run results

**Files:**
- Modify: `evals/runner.py` (`run_one` result dict ~line 202; `run_one_continual` result dict ~line 155)
- Test: `tests/test_eval_runner.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_eval_runner.py`:

```python
def test_run_result_reports_neuron_count():
    from evals.spec import SuiteSpec
    spec = SuiteSpec(variants=("core",), seeds=1, steps=20, record_every=10,
                     layers=(2, 3, 2), n_points=40)
    res = runner.run_one("core", 0, spec)
    assert res["n_neurons"] == 2 + 3 + 2
```

(If `tests/test_eval_runner.py` imports the runner differently, match its existing pattern — it already calls `run_one`.)

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_eval_runner.py::test_run_result_reports_neuron_count -q`
Expected: FAIL — `KeyError: 'n_neurons'`.

- [ ] **Step 3: Add `n_neurons` to both result dicts**

In `evals/runner.py`, in `run_one`'s returned dict add `"n_neurons": net.num_neurons,` (e.g. after `"initial_edge_count": len(initial_edges),`). Add the identical line to `run_one_continual`'s returned dict.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_eval_runner.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add evals/runner.py tests/test_eval_runner.py
git commit -m "runner: carry n_neurons on run results (scaling-chart x-axis)"
```

---

### Task 6: `_plot_scaling` chart in `evals/report.py`

**Files:**
- Modify: `evals/report.py` (`_plot_scaling`; call it in `write_report`)
- Test: `tests/test_eval_report.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_eval_report.py` (use `tmp_path`):

```python
def _agg_with_cost():
    def cell(m, s):
        return {"mean": m, "std": s, "ci_low": m, "ci_high": m, "verdict": ""}
    return {
        "variants": ["size-w4", "size-w16"],
        "baseline": "size-w16",
        "directions": {"ghost_dense_cost": "neutral",
                       "ghost_pairs_scored": "neutral"},
        "metrics": {
            "ghost_dense_cost": {"size-w4": cell(60, 0), "size-w16": cell(900, 0)},
            "ghost_pairs_scored": {"size-w4": cell(12, 0), "size-w16": cell(40, 0)},
        },
    }


def test_plot_scaling_writes_png_when_cost_present(tmp_path):
    results = [{"variant": "size-w4", "n_neurons": 14, "series": {}, "dist": {}},
               {"variant": "size-w16", "n_neurons": 50, "series": {}, "dist": {}}]
    path = tmp_path / "cost_scaling.png"
    ok = report._plot_scaling(_agg_with_cost(), results, str(path))
    assert ok is True and path.exists()


def test_plot_scaling_skips_when_cost_absent(tmp_path):
    agg = {"variants": ["a"], "baseline": "a", "directions": {},
           "metrics": {"final_test_acc": {"a": {"mean": 1.0, "std": 0.0}}}}
    path = tmp_path / "cost_scaling.png"
    ok = report._plot_scaling(agg, [{"variant": "a", "n_neurons": 5}], str(path))
    assert ok is False and not path.exists()
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_eval_report.py -k plot_scaling -q`
Expected: FAIL — `report._plot_scaling` does not exist.

- [ ] **Step 3: Implement `_plot_scaling` and wire it in**

In `evals/report.py`, add after `_plot_verdict_heatmap`:

```python
def _plot_scaling(agg, results, path) -> bool:
    """grow-scan cost vs network size: dense candidates (~N²) vs scored (sparse).

    Returns True if it wrote a chart, False if the cost metrics / neuron counts
    aren't present (caller skips silently).
    """
    m = agg.get("metrics", {})
    if "ghost_dense_cost" not in m or "ghost_pairs_scored" not in m:
        return False
    by = _by_variant(results)
    pts = []
    for v, runs in by.items():
        n = runs[0].get("n_neurons")
        dense = m["ghost_dense_cost"].get(v, {}).get("mean")
        scored = m["ghost_pairs_scored"].get(v, {}).get("mean")
        if n is None or dense is None or scored is None:
            continue
        pts.append((n, dense, scored))
    if len(pts) < 2:
        return False
    pts.sort()
    xs = [p[0] for p in pts]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, [p[1] for p in pts], "o-", label="dense candidates (≈N²)")
    ax.plot(xs, [p[2] for p in pts], "s-", label="scored (activity-sparse)")
    ax.set_title("grow-scan cost vs network size")
    ax.set_xlabel("neuron count (N)")
    ax.set_ylabel("candidate ghost wires per sample")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return True
```

In `write_report`, after the `_plot_verdict_heatmap(...)` call:

```python
    _plot_scaling(agg, results, os.path.join(out_dir, "cost_scaling.png"))
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_eval_report.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add evals/report.py tests/test_eval_report.py
git commit -m "report: cost-vs-size scaling chart (dense vs scored candidates)"
```

---

### Task 7: Add cost metrics to `KEY_METRICS`

**Files:**
- Modify: `evals/publish.py` (`KEY_METRICS`)
- Test: `tests/test_eval_publish.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_eval_publish.py`:

```python
def test_cost_metrics_are_key_metrics():
    from evals.publish import KEY_METRICS
    assert "ghost_dense_cost" in KEY_METRICS
    assert "ghost_pairs_scored" in KEY_METRICS
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_eval_publish.py::test_cost_metrics_are_key_metrics -q`
Expected: FAIL — not in `KEY_METRICS`.

- [ ] **Step 3: Add them**

In `evals/publish.py` `KEY_METRICS`, add a line after `"synapse_count_end", "effective_density",`:

```python
    "ghost_dense_cost", "ghost_pairs_scored",               # grow-scan compute cost
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_eval_publish.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add evals/publish.py tests/test_eval_publish.py
git commit -m "publish: surface grow-scan cost metrics in the key-metrics table"
```

---

### Task 8: `currency-bounded` variant in `evals/spec.py`

**Files:**
- Modify: `evals/spec.py` (`VARIANTS`)
- Test: `tests/test_eval_spec.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_eval_spec.py`:

```python
def test_currency_bounded_is_currency_plus_demand_k():
    from evals.spec import make_config
    base = make_config("currency")
    bounded = make_config("currency-bounded")
    assert base.grow_demand_k is None
    assert isinstance(bounded.grow_demand_k, int) and bounded.grow_demand_k > 0
    # identical to currency except for the demand bound
    assert bounded.grad_currency and bounded.enable_grow and bounded.enable_prune
    assert bounded.grow_bar_frac == base.grow_bar_frac
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_eval_spec.py::test_currency_bounded_is_currency_plus_demand_k -q`
Expected: FAIL — `KeyError: 'currency-bounded'`.

- [ ] **Step 3: Add the variant**

In `evals/spec.py` `VARIANTS`, add after the `"currency-gb3-ghost"` block (before the width-sweep block):

```python
    # === Phase-2 demand-gated grow scan =====================================
    # The currency baseline, but the grow scan scores ghosts only into the top-k
    # highest-|delta| post neurons (Config.grow_demand_k) — the bounded tier that
    # pushes scan cost toward ∝ active edges. Opt-in; validated for accuracy ≈
    # baseline before it could ever be promoted. k=4 is the moderate setting.
    "currency-bounded": lambda: Config(
        eta_base=0.02, grad_currency=True, enable_confidence=True,
        enable_prune=True, enable_grow=True,
        gamma_dec=0.001, t_struct=200, grow_demand_k=4),
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_eval_spec.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add evals/spec.py tests/test_eval_spec.py
git commit -m "spec: currency-bounded variant (demand-gated grow scan, k=4)"
```

---

### Task 9: Research roadmap section in the architecture doc

**Files:**
- Modify: `docs/v1_implementation.MD` (append a roadmap section)

- [ ] **Step 1: Add the section**

Append to `docs/v1_implementation.MD`:

```markdown
## Research roadmap: compute efficiency

The grow scan's `O(N²)` ghost enumeration was replaced by an activity-sparse scan
(see `sprout/currency.py` `active_ghost_sets` / `iter_ghost_candidates` and the
`docs/eval-runs` scaling run). Two further sparsity wins are deferred — both are
**constant-factor** (the per-step path is already `O(live edges)`, not `O(N²)`),
so they improve wall-clock but not the asymptotic scaling the grow-scan fix
addressed:

- **#2 Lazy gradient meters.** `update_gradient_meters` EMA-updates *all* live
  wires every step, but a wire whose pre was silent has gradient 0, so its update
  is pure decay `M ← β·M`. Track each wire's last-touched step and apply `β^Δ`
  lazily on next access, so per-step work touches only the active wires.
  Bit-identical to the current meters.
- **#3 Event-driven forward/backward.** `forward`/`backward` visit every hidden
  neuron each step. An active-frontier formulation propagates only out of neurons
  that fired (ReLU output ≠ 0), so silent neurons and their wires cost nothing —
  turning the per-step path from `∝ edges` into `∝ active edges`. The largest
  inner-loop change; left for when per-step wall-clock becomes the bottleneck.
```

- [ ] **Step 2: Commit**

```bash
git add docs/v1_implementation.MD
git commit -m "docs: research roadmap for deferred compute-efficiency work (#2, #3)"
```

---

### Task 10: Full suite green + run & publish the eval

**Files:** none (verification + publish)

- [ ] **Step 1: Run the full test suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass (was 216; this plan adds ~16 tests → ~232).

- [ ] **Step 2: Publish the scaling run (headline)**

```bash
.venv/bin/python evaluate.py --variants size-w4,size-w6,size-w10,size-w16,size-w24 \
  --baseline size-w16 --seeds 5 --dataset spirals --steps 15000 \
  --jobs 6 --no-cache --publish --run-name grow-scan-scaling
git add docs/eval-runs/grow-scan-scaling && git commit -m "eval: grow-scan-scaling (dense vs scored cost across width)" && git push
```

- [ ] **Step 3: Publish the demand-gating tradeoff run**

```bash
.venv/bin/python evaluate.py --variants currency,currency-bounded \
  --baseline currency --seeds 5 --dataset spirals --steps 15000 \
  --jobs 6 --no-cache --publish --run-name grow-scan-demand-gating
git add docs/eval-runs/grow-scan-demand-gating && git commit -m "eval: grow-scan-demand-gating (currency vs k=4 bound)" && git push
```

- [ ] **Step 4: Summarise** both runs to the user with the key-metrics table, plain verdict, the `docs/eval-runs/<name>/README.md` paths, and honest wins **and** losses (per the running-sprout-evals protocol).

---

## Self-review

- **Spec coverage:** §3.1 helpers → T1; §3.2 exact-sparse → T2; §3.3 `grow_demand_k` → T3; §3.4 threading → T3; §4 metrics → T4, chart → T6, `n_neurons` → T5, KEY_METRICS → T7; §4.1 runs → T10; §3.3 variant → T8; §2 roadmap → T9; §5 tests distributed across T1–T8 (bit-identity T2, `!=0` T2, demand-gating T2/T1, dense count T1/T4, metric registration T4, config default T3). All covered.
- **Placeholder scan:** none — every code step has complete code; `k=4` is a concrete chosen value.
- **Type/name consistency:** `active_ghost_sets`, `iter_ghost_candidates`, `dense_ghost_count`, `ghost_scan_cost`, `grow_demand_k`, `ghost_dense_cost`, `ghost_pairs_scored`, `_plot_scaling`, `currency-bounded`, `n_neurons` — used identically across all tasks.
