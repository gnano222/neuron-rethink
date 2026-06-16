# Conv Front-End (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether fixed translation-invariant features break SPROUT's ~0.93 14×14-MNIST ceiling, as one new dataset variant with zero changes to the network/economy.

**Architecture:** A fixed bank of 3×3 filters → ReLU → 2×2 max-pool → flatten → standardize, exposed as `mnist-conv`/`mnist-conv-rand` datasets. A per-variant `init_dataset` override lets raw vs conv arms share one eval suite for bootstrap-CI verdicts.

**Tech Stack:** Pure NumPy; existing `evals/` harness; pytest; `.venv/bin/python`.

---

### Task 1: `sprout/conv_features.py` — fixed filter bank + conv/pool

**Files:**
- Create: `sprout/conv_features.py`
- Test: `tests/test_conv_features.py`

- [ ] **Step 1: Write failing tests**

```python
import numpy as np
from sprout.conv_features import filter_bank, conv_features


def test_hand_bank_count_shape_norm():
    bank = filter_bank("hand")
    assert len(bank) == 6
    for k in bank:
        assert k.shape == (3, 3)
        assert abs(np.linalg.norm(k) - 1.0) < 1e-9

def test_random_bank_deterministic_and_zero_mean():
    a = filter_bank("random", seed=0)
    b = filter_bank("random", seed=0)
    assert len(a) == 6 and all(np.allclose(x, y) for x, y in zip(a, b))
    c = filter_bank("random", seed=1)
    assert not np.allclose(a[0], c[0])
    for k in a:
        assert abs(k.mean()) < 1e-9 and abs(np.linalg.norm(k) - 1.0) < 1e-9

def test_unknown_bank_raises():
    import pytest
    with pytest.raises(ValueError):
        filter_bank("nope")

def test_conv_features_shape_14x14_hand():
    imgs = np.random.default_rng(0).normal(size=(5, 14, 14))
    feats = conv_features(imgs, filter_bank("hand"), pool=2, nonlin="relu")
    # valid 3x3 -> 12x12, 2x2 pool -> 6x6=36, x6 filters = 216
    assert feats.shape == (5, 216)

def test_relu_nonneg():
    imgs = np.random.default_rng(1).normal(size=(3, 14, 14))
    feats = conv_features(imgs, filter_bank("random", seed=0), pool=2, nonlin="relu")
    assert (feats >= 0).all()

def test_maxpool_translation_invariance():
    # one bright pixel anywhere inside a single 2x2 pool window after a
    # delta/identity filter gives the SAME pooled value -> position-invariant.
    bank = [np.array([[0, 0, 0], [0, 1.0, 0], [0, 0, 0]])]  # identity (center tap)
    base = np.zeros((1, 14, 14)); base[0, 5, 5] = 3.0
    shift = np.zeros((1, 14, 14)); shift[0, 5, 6] = 3.0     # +1 col, same pool cell
    fb = conv_features(base, bank, pool=2, nonlin="relu")
    fs = conv_features(shift, bank, pool=2, nonlin="relu")
    assert np.allclose(fb, fs)
```

- [ ] **Step 2: Run, verify fail**

Run: `.venv/bin/python -m pytest tests/test_conv_features.py -q`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

```python
"""Fixed convolutional feature front-end (Phase 1 of conv/weight-sharing).

A *fixed* (untrained) bank of small filters turns a raw image into translation-
tolerant feature maps, which SPROUT then classifies as if they were the input.
Because the filters never learn, this is pure preprocessing: the network, the
currency economy, and the array backend are untouched. The point is to MEASURE
how much translation invariance buys before building the faithful weight-sharing
version (Phase 2). See docs/superpowers/specs/2026-06-16-conv-weight-sharing-design.md.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

_EPS = 1e-12


def _hand_bank():
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)  # vertical edge
    sy = sx.T                                                          # horizontal edge
    d1 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=float)  # 45 deg
    d2 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype=float)  # 135 deg
    center = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=float)  # blob on
    surround = -center                                               # blob off
    return [sx, sy, d1, d2, center, surround]


def filter_bank(kind="hand", seed=0, size=3):
    """Return a list of ``(size,size)`` ℓ2-normalized kernels.

    ``hand``: 4 oriented edges + 2 center-surround blobs (6 filters; ``size`` must
    be 3). ``random``: 6 zero-mean, unit-norm Gaussian kernels (fixed by ``seed``).
    """
    if kind == "hand":
        if size != 3:
            raise ValueError("hand bank is defined for size=3 only")
        bank = _hand_bank()
    elif kind == "random":
        rng = np.random.default_rng(seed)
        bank = []
        for _ in range(6):
            k = rng.normal(size=(size, size))
            bank.append(k - k.mean())
        bank = [k for k in bank]
    else:
        raise ValueError(f"unknown filter bank kind {kind!r}")
    return [k / (np.linalg.norm(k) + _EPS) for k in bank]


def _nonlin(x, kind):
    if kind == "relu":
        return np.maximum(x, 0.0)
    if kind == "abs":
        return np.abs(x)
    raise ValueError(f"unknown nonlin {kind!r}")


def _maxpool(a, pool):
    """Non-overlapping 2D max-pool over the last two axes (drops a ragged edge)."""
    n, h, w = a.shape
    hc, wc = (h // pool) * pool, (w // pool) * pool
    a = a[:, :hc, :wc].reshape(n, hc // pool, pool, wc // pool, pool)
    return a.max(axis=(2, 4))


def conv_features(images, bank, pool=2, nonlin="relu"):
    """Apply each fixed kernel (valid correlation), nonlinearity, and max-pool,
    then flatten + concatenate across filters.

    ``images``: ``(N, H, W)``. Returns ``(N, n_filters * pooled_h * pooled_w)``.
    """
    images = np.asarray(images, dtype=float)
    feats = []
    for k in bank:
        kh, kw = k.shape
        win = sliding_window_view(images, (kh, kw), axis=(1, 2))   # (N,oh,ow,kh,kw)
        conv = np.einsum("nijkl,kl->nij", win, k)
        feats.append(_maxpool(_nonlin(conv, nonlin), pool).reshape(len(images), -1))
    return np.concatenate(feats, axis=1)
```

- [ ] **Step 4: Run, verify pass**

Run: `.venv/bin/python -m pytest tests/test_conv_features.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add sprout/conv_features.py tests/test_conv_features.py
git commit -m "feat: fixed conv feature front-end (filter_bank + conv_features)"
```

---

### Task 2: conv MNIST loaders + registry (datasets.py)

**Files:**
- Modify: `sprout/datasets.py`
- Test: `tests/test_datasets.py`

- [ ] **Step 1: Write failing tests** (pure transform — no network)

```python
def test_mnist_conv_transform_shape_and_invariance():
    from sprout.datasets import mnist_conv_transform
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4, 196))          # 4 fake 14x14 images, flat
    F = mnist_conv_transform(X, side=14, bank_kind="hand", pool=2, nonlin="relu")
    assert F.shape == (4, 216)

def test_get_dataset_unknown_still_raises_after_conv():
    import pytest
    from sprout.datasets import get_dataset
    with pytest.raises(ValueError):
        get_dataset("mnist-conv-nope", seed=0)
```

Plus a network-gated end-to-end (mirrors the existing skip pattern):

```python
def test_get_dataset_mnist_conv_is_216():
    pytest = __import__("pytest")
    try:
        from sprout.datasets import get_dataset
        Xtr, ytr, Xte, yte = get_dataset("mnist-conv", seed=0, n_points=200)
    except Exception as e:
        pytest.skip(f"MNIST fetch unavailable: {e}")
    assert Xtr.shape[1] == 216 and len(Xtr) == 200 and len(Xte) == 1000
    assert np.allclose(Xtr.mean(axis=0), 0.0, atol=1e-6)   # standardized on train
```

- [ ] **Step 2: Run, verify fail**

Run: `.venv/bin/python -m pytest tests/test_datasets.py -q -k conv`
Expected: FAIL (no `mnist_conv_transform`).

- [ ] **Step 3: Implement** — add to `sprout/datasets.py`:

```python
def mnist_conv_transform(X_flat, side, bank_kind="hand", pool=2, nonlin="relu"):
    """Flat (N, side*side) images -> (N, n_filters*pooled) fixed-conv features.
    Pure preprocessing (no training); the bank is fixed (seed 0)."""
    from sprout.conv_features import filter_bank, conv_features
    imgs = np.asarray(X_flat, dtype=float).reshape(-1, side, side)
    return conv_features(imgs, filter_bank(bank_kind, seed=0), pool=pool, nonlin=nonlin)


def load_mnist_conv_split(seed=0, n_train=12000, n_test=1000, downsample=True,
                          bank_kind="hand", pool=2, nonlin="relu"):
    """MNIST passed through a FIXED conv feature front-end, then standardized on
    TRAIN stats. ``downsample`` 28->14 first (side=14, else 28)."""
    from sklearn.datasets import fetch_openml
    data = fetch_openml("mnist_784", version=1, as_frame=False,
                        parser="liac-arff", cache=True)
    X = np.asarray(data.data, dtype=float)
    side = 28
    if downsample:
        X = _downsample_2x2(X)
        side = 14
    y = np.asarray(data.target, dtype=int)
    Xtr, ytr, Xte, yte = _stratified_subsample_split(X, y, n_train, n_test, seed)
    Ftr = mnist_conv_transform(Xtr, side, bank_kind, pool, nonlin)
    Fte = mnist_conv_transform(Xte, side, bank_kind, pool, nonlin)
    Ftr, Fte = _standardize_on_train(Ftr, Fte)
    return Ftr, ytr, Fte, yte
```

And in `get_dataset`, before the final `raise`:

```python
    if name == "mnist-conv":                            # 14x14 -> hand-filter conv
        return load_mnist_conv_split(seed=seed, n_train=n_points, n_test=1000,
                                     downsample=True, bank_kind="hand")
    if name == "mnist-conv-rand":                       # 14x14 -> random-filter conv
        return load_mnist_conv_split(seed=seed, n_train=n_points, n_test=1000,
                                     downsample=True, bank_kind="random")
    if name == "mnist-full-conv":                       # 28x28 -> hand-filter conv
        return load_mnist_conv_split(seed=seed, n_train=n_points, n_test=1000,
                                     downsample=False, bank_kind="hand")
```

- [ ] **Step 4: Run, verify pass**

Run: `.venv/bin/python -m pytest tests/test_datasets.py -q`
Expected: PASS (network-gated test skips if offline).

- [ ] **Step 5: Commit**

```bash
git add sprout/datasets.py tests/test_datasets.py
git commit -m "feat: conv MNIST datasets (mnist-conv / mnist-conv-rand / mnist-full-conv)"
```

---

### Task 3: per-variant dataset override + conv variants

**Files:**
- Modify: `sprout/train.py` (Config), `evals/runner.py`, `evals/cli.py`, `evals/spec.py`
- Test: `tests/test_eval_runner.py`, `tests/test_eval_spec.py`

- [ ] **Step 1: Write failing tests**

`tests/test_eval_runner.py`:

```python
def test_dataset_name_override():
    from evals.runner import _dataset_name
    from evals.spec import SuiteSpec
    from sprout.train import Config
    spec = SuiteSpec(dataset="spirals")
    assert _dataset_name(Config(), spec) == "spirals"            # no override
    assert _dataset_name(Config(init_dataset="mnist-conv"), spec) == "mnist-conv"
```

`tests/test_eval_spec.py`:

```python
def test_conv_variants_carry_dataset_and_input_size():
    from evals.spec import make_config
    for name, ds in (("mnist-conv-hand", "mnist-conv"),
                     ("mnist-conv-rand", "mnist-conv-rand")):
        cfg = make_config(name)
        assert cfg.init_dataset == ds
        assert cfg.init_layers[0] == 216 and cfg.init_layers[-1] == 10
        assert cfg.grow_demand_k == 4          # in _BOUNDED_GROW_VARIANTS
```

- [ ] **Step 2: Run, verify fail**

Run: `.venv/bin/python -m pytest tests/test_eval_runner.py::test_dataset_name_override tests/test_eval_spec.py::test_conv_variants_carry_dataset_and_input_size -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

In `sprout/train.py` `Config`, beside `init_layers`:

```python
    # Optional per-variant override of the eval dataset (sibling to init_layers/
    # init_density): lets one suite mix datasets — e.g. raw vs conv-feature MNIST —
    # so they share bootstrap-CI verdicts. Trainer ignores it; only the runner reads it.
    init_dataset: str | None = None
```

In `evals/runner.py`, add the helper and use it in `run_one`:

```python
def _dataset_name(cfg, spec) -> str:
    """Dataset for this variant: its own ``init_dataset`` override if set, else the
    suite-wide ``spec.dataset``. Lets raw and conv-feature arms share one suite."""
    override = getattr(cfg, "init_dataset", None)
    return spec.dataset if override is None else override
```

Then in `run_one`, replace the `get_dataset(spec.dataset, ...)` call's first arg
with `_dataset_name(cfg, spec)`.

In `evals/cli.py`, extend `--dataset` choices:

```python
                    choices=["spirals", "blobs", "digits", "mnist", "mnist-full",
                             "mnist-conv", "mnist-conv-rand", "mnist-full-conv"])
```

In `evals/spec.py`, give `_sparse`/`_dense` an optional `dataset`:

```python
def _dense(layers, dataset=None):
    return lambda: Config(eta_base=0.02, init_density=1.0, init_layers=layers,
                          init_dataset=dataset)


def _sparse(layers, density, k=4, dataset=None):
    return lambda: Config(
        eta_base=0.02, grad_currency=True, enable_confidence=True,
        enable_prune=True, enable_grow=True, gamma_dec=0.001, t_struct=200,
        phasic_structure=True, startle=True, grow_demand_k=k,
        init_layers=layers, init_density=density, init_dataset=dataset)
```

Add the two conv variants (next to the mnist arms):

```python
    # --- CONV front-end (Phase 1 measurement): same w32-sparse architecture as
    # mnist-w32-sparse, but the input is FIXED-conv features (216) not raw pixels
    # (196). Run in one suite with mnist-w32-sparse (baseline, raw mnist) to test
    # whether translation-invariant features break the ~0.93 14x14 ceiling.
    "mnist-conv-hand": _sparse((216, 32, 10), 0.5, dataset="mnist-conv"),
    "mnist-conv-rand": _sparse((216, 32, 10), 0.5, dataset="mnist-conv-rand"),
```

Add `"mnist-conv-hand"`, `"mnist-conv-rand"` to `_BOUNDED_GROW_VARIANTS`.

- [ ] **Step 4: Run, verify pass + no regressions**

Run: `.venv/bin/python -m pytest tests/test_eval_runner.py tests/test_eval_spec.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sprout/train.py evals/runner.py evals/cli.py evals/spec.py tests/test_eval_runner.py tests/test_eval_spec.py
git commit -m "feat: per-variant dataset override + conv-front-end eval variants"
```

---

### Task 4: full suite + the measurement run

- [ ] **Step 1: Full test suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass (was 219 + new tests).

- [ ] **Step 2: The conv-vs-raw measurement (5 seeds, 14×14)**

```bash
.venv/bin/python evaluate.py \
  --variants mnist-w32-sparse,mnist-conv-hand,mnist-conv-rand \
  --seeds 5 --dataset mnist --steps 15000 --baseline mnist-w32-sparse \
  --points 12000 --train-eval-cap 2000 --record-every 1000 \
  --jobs 6 --no-cache --publish --run-name conv-front-end-mnist14
```

- [ ] **Step 3: Commit + push the run folder**

```bash
git add docs/eval-runs/conv-front-end-mnist14
git commit -m "eval: conv front-end vs raw on 14x14 MNIST"
git push
```

- [ ] **Step 4: 784 follow-up — only if 14×14 shows a ▲**

```bash
.venv/bin/python evaluate.py \
  --variants mnist784-d1-sparse,mnist-full-conv-hand \
  --seeds 5 --dataset mnist-full --steps 15000 --baseline mnist784-d1-sparse \
  --points 12000 --train-eval-cap 2000 --record-every 1000 \
  --jobs 6 --no-cache --publish --run-name conv-front-end-mnist784
```

(Add a `mnist-full-conv-hand` variant `_sparse((1014,32,10),0.25,dataset="mnist-full-conv")` only if pursuing this; 28×28 hand-conv = 6×13×13 = 1014 features.)

---

### Task 5: summarise findings

- [ ] Write the chat summary: key-metrics table (▲/▼/≈ with the "What it means"
  column), 2–3 sentence verdict, the `docs/eval-runs/conv-front-end-mnist14/README.md`
  path, honest wins AND losses, and the **Phase-2 go/no-go decision**.
- [ ] Update memory (`sprout-digits-adaptation.md` or a new conv note + MEMORY.md).
```
