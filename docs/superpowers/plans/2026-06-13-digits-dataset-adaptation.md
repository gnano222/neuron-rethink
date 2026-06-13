# Digits Dataset Adaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the gradient-as-currency SPROUT architecture to scikit-learn's 8×8 digit dataset (64 features, 10 classes), staged smoke-test → registry/harness integration, keeping the sparse-vs-fully-connected compute-efficiency story.

**Architecture:** A new `sprout/datasets.py` seam holds the digits loader (standardize on train stats + seeded stratified split) and a `get_dataset` registry that unifies generated (spirals/blobs, byte-identical to today) and fixed (digits) datasets. `run_digits.py` proves learning first; the eval runner/CLI then consume the registry. The core network needs no changes (forward/backward already K-class general).

**Tech Stack:** Python, NumPy, scikit-learn (`load_digits` only), existing SPROUT modules.

---

### Task 0: Confirm scikit-learn availability

- [ ] **Step 1:** Run `python -c "import sklearn; from sklearn.datasets import load_digits; d=load_digits(); print(d.data.shape, d.target.min(), d.target.max())"`
  Expected: `(1797, 64) 0 9`. If `ModuleNotFoundError`, run `pip install scikit-learn` and add `scikit-learn` to the project's requirements/deps file, then re-run.

---

### Task 1: Digits loader + stratified split (`sprout/datasets.py`)

**Files:**
- Create: `sprout/datasets.py`
- Test: `tests/test_datasets.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_datasets.py
import numpy as np
from sprout.datasets import load_digits_split


def test_digits_shapes_and_labels():
    Xtr, ytr, Xte, yte = load_digits_split(seed=0, test_frac=0.2)
    assert Xtr.shape[1] == 64 and Xte.shape[1] == 64
    assert len(Xtr) == len(ytr) and len(Xte) == len(yte)
    assert len(Xtr) + len(Xte) == 1797
    assert set(np.unique(ytr)).issubset(set(range(10)))
    assert set(np.unique(yte)) == set(range(10))     # all classes in test
    assert ytr.dtype.kind in "iu" and yte.dtype.kind in "iu"


def test_digits_deterministic():
    a = load_digits_split(seed=3)
    b = load_digits_split(seed=3)
    for u, v in zip(a, b):
        assert np.array_equal(u, v)
    c = load_digits_split(seed=4)
    assert not np.array_equal(a[0], c[0])            # different seed -> different split


def test_digits_standardized_on_train_no_leakage():
    Xtr, ytr, Xte, yte = load_digits_split(seed=0)
    # train columns are ~zero-mean / unit-var (constant columns stay ~0)
    assert np.allclose(Xtr.mean(axis=0), 0.0, atol=1e-6)
    stds = Xtr.std(axis=0)
    assert np.all(stds <= 1.0 + 1e-6)
    # test set is NOT forced to zero mean (proves train stats were reused)
    assert not np.allclose(Xte.mean(axis=0), 0.0, atol=1e-3)


def test_digits_stratified_proportions():
    Xtr, ytr, Xte, yte = load_digits_split(seed=1, test_frac=0.2)
    for c in range(10):
        frac = (yte == c).sum() / ((ytr == c).sum() + (yte == c).sum())
        assert 0.1 < frac < 0.3                      # ~0.2 each class
```

- [ ] **Step 2: Run to verify they fail**
  Run: `python -m pytest tests/test_datasets.py -q`  Expected: FAIL (`No module named sprout.datasets`).

- [ ] **Step 3: Implement**

```python
# sprout/datasets.py
"""Dataset registry for SPROUT.

Unifies *generated* toy datasets (spirals/blobs, reproduced byte-for-byte from
sprout.data so the eval cache and pinned baselines are undisturbed) and *fixed*
datasets (scikit-learn 8x8 digits) behind one ``get_dataset`` interface that
returns ``(X_tr, y_tr, X_te, y_te)``. New fixed datasets (e.g. MNIST) plug in
here as one extra loader branch; the eval harness never changes.
"""

from __future__ import annotations

import numpy as np

from sprout.data import generate_blobs, generate_spirals

_EPS = 1e-8


def _stratified_split(X, y, test_frac, seed):
    """Deterministic per-class shuffle + split (no sklearn.model_selection dep)."""
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_frac))
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())
    train_idx = np.array(sorted(train_idx))
    test_idx = np.array(sorted(test_idx))
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def load_digits_split(seed: int = 0, test_frac: float = 0.2):
    """scikit-learn 8x8 digits: seeded stratified split, standardized on TRAIN
    statistics only (no leakage; the eps guard handles constant corner pixels)."""
    from sklearn.datasets import load_digits
    data = load_digits()
    X = np.asarray(data.data, dtype=float)        # (1797, 64), pixels 0..16
    y = np.asarray(data.target, dtype=int)        # labels 0..9
    Xtr, ytr, Xte, yte = _stratified_split(X, y, test_frac, seed)
    mu = Xtr.mean(axis=0)
    sigma = Xtr.std(axis=0)
    Xtr = (Xtr - mu) / (sigma + _EPS)
    Xte = (Xte - mu) / (sigma + _EPS)
    return Xtr, ytr, Xte, yte
```

- [ ] **Step 4: Run to verify pass**
  Run: `python -m pytest tests/test_datasets.py -q`  Expected: PASS (4 tests).

- [ ] **Step 5: Commit**
  `git add sprout/datasets.py tests/test_datasets.py && git commit -m "feat: digits loader with standardized stratified split"` (+ Co-Authored-By trailer).

---

### Task 2: `get_dataset` registry (byte-identical generated path)

**Files:**
- Modify: `sprout/datasets.py`
- Test: `tests/test_datasets.py`

- [ ] **Step 1: Write failing tests**

```python
# append to tests/test_datasets.py
from sprout.data import generate_spirals, generate_blobs
from sprout.datasets import get_dataset


def test_get_dataset_spirals_byte_identical():
    Xtr, ytr, Xte, yte = get_dataset("spirals", seed=2, n_points=600,
                                     turns=1.0, noise=0.10, test_seed_offset=10000)
    eXtr, eytr = generate_spirals(n=600, seed=2, turns=1.0, noise=0.10)
    eXte, eyte = generate_spirals(n=600, seed=10002, turns=1.0, noise=0.10)
    assert np.array_equal(Xtr, eXtr) and np.array_equal(ytr, eytr)
    assert np.array_equal(Xte, eXte) and np.array_equal(yte, eyte)


def test_get_dataset_blobs_byte_identical():
    Xtr, ytr, Xte, yte = get_dataset("blobs", seed=5, n_points=600,
                                     turns=1.0, noise=0.10, test_seed_offset=10000)
    eXtr, eytr = generate_blobs(n=600, seed=5)
    eXte, eyte = generate_blobs(n=600, seed=10005)
    assert np.array_equal(Xtr, eXtr) and np.array_equal(Xte, eXte)


def test_get_dataset_digits_tuple():
    Xtr, ytr, Xte, yte = get_dataset("digits", seed=0)
    assert Xtr.shape[1] == 64 and set(np.unique(yte)) == set(range(10))
```

- [ ] **Step 2: Run to verify fail**
  Run: `python -m pytest tests/test_datasets.py -q`  Expected: FAIL (`cannot import name get_dataset`).

- [ ] **Step 3: Implement (append to `sprout/datasets.py`)**

```python
def get_dataset(name, seed, *, n_points=600, turns=1.0, noise=0.10,
                test_seed_offset=10000):
    """Return (X_tr, y_tr, X_te, y_te) for a named dataset.

    Generated datasets draw train at ``seed`` and test at
    ``seed + test_seed_offset`` with the exact args the runner used before, so
    spirals/blobs are byte-identical (eval cache + pinned baselines safe).
    Fixed datasets (digits) ignore the generator knobs.
    """
    if name == "digits":
        return load_digits_split(seed=seed)
    if name == "blobs":
        Xtr, ytr = generate_blobs(n=n_points, seed=seed)
        Xte, yte = generate_blobs(n=n_points, seed=seed + test_seed_offset)
        return Xtr, ytr, Xte, yte
    if name == "spirals":
        Xtr, ytr = generate_spirals(n=n_points, seed=seed, turns=turns, noise=noise)
        Xte, yte = generate_spirals(n=n_points, seed=seed + test_seed_offset,
                                    turns=turns, noise=noise)
        return Xtr, ytr, Xte, yte
    raise ValueError(f"unknown dataset {name!r}")
```

- [ ] **Step 4: Run to verify pass**
  Run: `python -m pytest tests/test_datasets.py -q`  Expected: PASS (7 tests).

- [ ] **Step 5: Commit**
  `git add sprout/datasets.py tests/test_datasets.py && git commit -m "feat: get_dataset registry (spirals/blobs byte-identical + digits)"`.

---

### Task 3: Stage-1 smoke-test (`run_digits.py`)

**Files:**
- Create: `run_digits.py`
- Test: `tests/test_run_digits.py`

- [ ] **Step 1: Write failing test** (tiny, fast — proves learning beats chance)

```python
# tests/test_run_digits.py
from run_digits import train_digits


def test_train_digits_beats_chance_quickly():
    res = train_digits(steps=1500, seed=0, layers=(64, 32, 16, 10),
                       density=0.4, record=False)
    assert res["test_acc"] > 0.30          # chance = 0.10; 1.5k steps clears it
    assert res["synapses"] < 64 * 32 + 32 * 16 + 16 * 10   # sparser than dense
    assert res["edge_steps"] > 0 and res["wall_time"] >= 0.0
```

- [ ] **Step 2: Run to verify fail**
  Run: `python -m pytest tests/test_run_digits.py -q`  Expected: FAIL (`No module named run_digits`).

- [ ] **Step 3: Implement `run_digits.py`** (mirrors the promoted `phasic-startle-k4` config)

```python
"""Stage-1 smoke test: train the SPROUT gradient-as-currency architecture on
scikit-learn 8x8 digits (10 classes) and report accuracy + compute cost.

    python run_digits.py --steps 40000 --seed 0 --layers 64,64,32,10
"""

from __future__ import annotations

import argparse
import time

from sprout.datasets import load_digits_split
from sprout.network import build_graph, init_weights
from sprout.train import Config, Trainer, accuracy


def _digits_config() -> Config:
    """The promoted phasic-startle-k4 architecture (matches the eval baseline)."""
    return Config(
        eta_base=0.02, enable_confidence=True, enable_prune=True, enable_grow=True,
        gamma_dec=0.001, t_struct=200, phasic_structure=True, startle=True,
        grow_demand_k=4,
    )


def train_digits(steps=40000, seed=0, layers=(64, 64, 32, 10), density=0.4,
                 record=True):
    Xtr, ytr, Xte, yte = load_digits_split(seed=seed)
    net = build_graph(list(layers), density=density, seed=seed)
    init_weights(net, seed=seed)
    cfg = _digits_config()
    tr = Trainer(cfg, net, Xtr, ytr, seed=seed)

    edge_steps = 0.0
    t0 = time.perf_counter()
    for _ in range(steps):
        tr.step(record=False)
        edge_steps += len(net.synapses)
    wall = time.perf_counter() - t0

    return {
        "test_acc": accuracy(net, Xte, yte),
        "train_acc": accuracy(net, Xtr, ytr),
        "synapses": len(net.synapses),
        "edge_steps": edge_steps,
        "wall_time": wall,
        "avg_live_edges": edge_steps / steps if steps else 0.0,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="SPROUT on 8x8 digits (smoke test)")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", default="64,64,32,10")
    ap.add_argument("--density", type=float, default=0.4)
    args = ap.parse_args(argv)
    layers = tuple(int(x) for x in args.layers.split(","))
    print(f"training {layers} @ density {args.density} for {args.steps} steps ...")
    r = train_digits(steps=args.steps, seed=args.seed, layers=layers,
                     density=args.density)
    print(f"  test acc      : {r['test_acc']:.4f}")
    print(f"  train acc     : {r['train_acc']:.4f}")
    print(f"  synapses      : {r['synapses']}")
    print(f"  avg live edges: {r['avg_live_edges']:.1f}")
    print(f"  edge-steps    : {r['edge_steps']:.3e}")
    print(f"  wall time     : {r['wall_time']:.1f}s "
          f"({1000 * r['wall_time'] / args.steps:.3f} ms/step)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**
  Run: `python -m pytest tests/test_run_digits.py -q`  Expected: PASS. If `test_acc` is borderline, bump the test's `steps` (not the threshold).

- [ ] **Step 5: Commit**
  `git add run_digits.py tests/test_run_digits.py && git commit -m "feat: run_digits.py Stage-1 smoke test"`.

---

### Task 4: Wire the eval runner to the registry

**Files:**
- Modify: `evals/runner.py` (`run_one`, imports; remove `_gen`)
- Test: `tests/test_eval_runner.py` (add a digits smoke + keep spirals green)

- [ ] **Step 1: Write failing test** (digits run produces a well-formed result)

```python
# add to tests/test_eval_runner.py
def test_run_one_digits_smoke():
    from evals.runner import run_one
    from evals.spec import SuiteSpec
    spec = SuiteSpec(variants=("phasic-startle-k4",), seeds=1, dataset="digits",
                     steps=400, shift_steps=0, record_every=200,
                     baseline="phasic-startle-k4", layers=(64, 32, 16, 10),
                     density=0.4)
    res = run_one("phasic-startle-k4", seed=0, spec=spec)
    assert res["regime"] == "single"
    assert 0.0 <= res["final"]["final_test_acc"] <= 1.0
    assert res["n_neurons"] == 64 + 32 + 16 + 10
```

- [ ] **Step 2: Run to verify fail**
  Run: `python -m pytest tests/test_eval_runner.py::test_run_one_digits_smoke -q`
  Expected: FAIL (runner's `_gen` raises/relabels for "digits" or shift path mismatches).

- [ ] **Step 3: Implement** — in `evals/runner.py`:
  - Replace the import line `from sprout.data import generate_blobs, generate_spirals` with `from sprout.data import generate_spirals` (still used by `run_one_continual`) and add `from sprout.datasets import get_dataset`.
  - Delete the `_gen` function.
  - In `run_one`, replace the two `_gen(...)` lines with:

```python
    X_tr, y_tr, X_te, y_te = get_dataset(
        spec.dataset, seed, n_points=spec.n_points, turns=spec.turns,
        noise=spec.noise, test_seed_offset=spec.test_seed_offset)
```

  - Guard the binary-only shift (just before the `if spec.shift_steps > 0:` body):

```python
    if spec.shift_steps > 0:
        if int(np.max(y_tr)) > 1:
            raise ValueError("label-swap shift is only defined for binary tasks")
        shift_start_index = len(series["rec_step"])
        y_tr_sw, y_te_final = 1 - y_tr, 1 - y_te
        ...
```

- [ ] **Step 4: Run to verify pass**
  Run: `python -m pytest tests/test_eval_runner.py -q`  Expected: PASS (digits smoke + all existing spiral runner tests still green — byte-identity from Task 2 guarantees no result drift).

- [ ] **Step 5: Commit**
  `git add evals/runner.py tests/test_eval_runner.py && git commit -m "feat: route eval runner through get_dataset; binary-only shift guard"`.

---

### Task 5: Add `digits` to the CLI dataset choices

**Files:**
- Modify: `evals/cli.py:35`
- Test: `tests/test_eval_cli.py`

- [ ] **Step 1: Write failing test**

```python
# add to tests/test_eval_cli.py
def test_cli_accepts_digits_dataset():
    from evals.cli import parse_args, build_spec
    args = parse_args(["--variants", "phasic-startle-k4,fully-connected",
                       "--dataset", "digits", "--layers", "64,64,32,10",
                       "--baseline", "phasic-startle-k4"])
    spec = build_spec(args)
    assert spec.dataset == "digits"
    assert spec.layers == (64, 64, 32, 10)
```

- [ ] **Step 2: Run to verify fail**
  Run: `python -m pytest tests/test_eval_cli.py::test_cli_accepts_digits_dataset -q`
  Expected: FAIL (argparse rejects `digits` — not in choices).

- [ ] **Step 3: Implement** — in `evals/cli.py`, change
  `ap.add_argument("--dataset", default="spirals", choices=["spirals", "blobs"])`
  to `choices=["spirals", "blobs", "digits"]`.

- [ ] **Step 4: Run to verify pass**
  Run: `python -m pytest tests/test_eval_cli.py -q`  Expected: PASS.

- [ ] **Step 5: Commit**
  `git add evals/cli.py tests/test_eval_cli.py && git commit -m "feat: --dataset digits in eval CLI"`.

---

### Task 6: Full regression + smoke verification

- [ ] **Step 1:** Run the whole suite: `python -m pytest -q`  Expected: all pass (prior 219 + the new tests).
- [ ] **Step 2:** Run the architecture guardrail: `python validate.py`  Expected: 7/7 (spirals path untouched — byte-identical).
- [ ] **Step 3:** Real smoke run: `python run_digits.py --steps 40000 --seed 0 --layers 64,64,32,10`  Expected: test acc ≥ ~0.85, synapses ≪ dense, ms/step printed. Record the numbers in the final report. If accuracy is low, tune steps/density/layers (capture the chosen values).
- [ ] **Step 4:** Final commit if any tuning changed defaults.

---

## Self-Review

- **Spec coverage:** Stage 1 → Task 3 (`run_digits.py` + sizing). Stage 3 registry → Tasks 1-2 (`sprout/datasets.py`), harness → Task 4 (runner), CLI → Task 5. Tests (registry/regression/learning/harness) → Tasks 1,2,3,4,6. Binary-shift guard → Task 4. ✅
- **Placeholders:** none — every step has concrete code/commands.
- **Type consistency:** `load_digits_split` / `get_dataset` / `train_digits` signatures are identical across the tasks that define and call them. `get_dataset` returns the 4-tuple consumed by `run_one`.
- **Out of scope (per spec):** MNIST, multiclass shift, homeostasis, >2D viz — none added. ✅
