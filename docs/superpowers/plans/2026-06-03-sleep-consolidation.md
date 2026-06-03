# Sleep consolidation — implementation plan

> **For agentic workers:** TDD task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add opt-in settledness-gated "sleep" consolidation — prune aggressively
only when an EMA-of-loss plateau says the net has settled.

**Architecture:** A `SettlednessDetector` (new `sprout/sleep.py`) wired into the
`Trainer` currency rewire path; consolidation reuses `prune_currency` with
aggressive params and pauses `grow`. Off by default. See the design spec
`docs/superpowers/specs/2026-06-03-sleep-consolidation-design.md`.

**Tech stack:** Python, numpy; pytest; the existing SPROUT currency code.

---

### Task 1: Config sleep fields

**Files:** Modify `sprout/train.py` (`Config`); Test `tests/test_train.py`.

- [ ] **Step 1 — failing test** (`tests/test_train.py`):
```python
def test_default_config_has_sleep_off():
    cfg = Config()
    assert cfg.enable_sleep is False
    assert cfg.sleep_warmup == 2500
    assert cfg.sleep_loss_beta == 0.01
    assert cfg.sleep_loss_tol == 0.03
    assert cfg.sleep_patience == 1500
    assert cfg.sleep_prune_floor == 2.0
    assert cfg.sleep_max_prune == 10
```
- [ ] **Step 2** — run, expect AttributeError.
- [ ] **Step 3** — add the 7 fields to `Config` (defaults as above).
- [ ] **Step 4** — run, expect PASS; run full `tests/test_train.py`.
- [ ] **Step 5** — commit.

### Task 2: SettlednessDetector

**Files:** Create `sprout/sleep.py`; Test `tests/test_sleep.py`.

- [ ] **Step 1 — failing tests** (`tests/test_sleep.py`):
```python
from sprout.sleep import SettlednessDetector

def test_not_settled_before_warmup():
    d = SettlednessDetector(beta=0.5, tol=0.01, patience=2, warmup=100)
    # feed a flat (plateaued) loss but stay under warmup
    assert not any(d.update(1.0, step) for step in range(50))

def test_settles_after_patience_on_plateau():
    d = SettlednessDetector(beta=0.5, tol=0.01, patience=3, warmup=0)
    # a strictly improving loss never settles
    assert not d.update(1.0, 0)
    assert not d.update(0.5, 1)
    assert not d.update(0.25, 2)
    # now flat: improvements stop -> after `patience` flat steps, settled
    flags = [d.update(0.25, s) for s in range(3, 9)]
    assert flags[-1] is True
    assert flags[0] is False  # not immediately

def test_new_improvement_resets_since_improve():
    d = SettlednessDetector(beta=1.0, tol=0.01, patience=2, warmup=0)
    d.update(1.0, 0)
    d.update(1.0, 1); d.update(1.0, 2)          # plateau builds
    assert d.update(0.5, 3) is False            # big improvement resets
    assert d.update(0.5, 4) is False            # rebuilding patience

def test_reset_requires_fresh_plateau():
    d = SettlednessDetector(beta=1.0, tol=0.01, patience=2, warmup=0)
    for s in range(5):
        d.update(1.0, s)
    assert d.update(1.0, 5) is True             # settled
    d.reset()
    assert d.update(1.0, 6) is False            # must re-plateau after reset
```
- [ ] **Step 2** — run, expect ImportError.
- [ ] **Step 3** — implement `SettlednessDetector` per the spec (EMA seed on first
  loss; `best`/`since_improve`; `update` returns `step>=warmup and
  since_improve>=patience`; `reset` zeroes `since_improve` and sets `best` to the
  current `loss_ema`).
- [ ] **Step 4** — run, expect PASS.
- [ ] **Step 5** — commit.

### Task 3: Trainer sleep wiring

**Files:** Modify `sprout/train.py` (`Trainer.__init__`, `step`,
`_step_currency`, `_rewire_currency`); Test `tests/test_train.py`.

- [ ] **Step 1 — failing test** (`tests/test_train.py`): an aggressive-sleep
  trainer fires a consolidation and ends sparser than a no-sleep twin.
```python
def test_sleep_consolidation_fires_and_sparsifies():
    from sprout.network import build_graph, init_weights
    from sprout.data import generate_spirals
    base = dict(eta_base=0.02, grad_currency=True, enable_confidence=True,
                enable_prune=True, enable_grow=True, gamma_dec=0.001, t_struct=100)
    X, y = generate_spirals(n=400, seed=0)

    def run(sleep):
        net = build_graph([2, 12, 12, 8, 2], density=0.5, seed=0)
        init_weights(net, seed=0)
        cfg = Config(**base, enable_sleep=sleep, sleep_warmup=500,
                     sleep_patience=300, sleep_loss_tol=0.05, sleep_max_prune=8)
        tr = Trainer(cfg, net, X, y, seed=0)
        for _ in range(4000):
            tr.step()
        return tr, len(net.synapses)

    tr_sleep, n_sleep = run(True)
    _, n_wake = run(False)
    assert any(e["type"] == "sleep" for e in tr_sleep.events)   # it slept
    assert n_sleep < n_wake                                      # and ended sparser
```
- [ ] **Step 2** — run, expect failure (no sleep events / not sparser).
- [ ] **Step 3** — implement: in `__init__` build `self.sleep_detector` +
  `self.settled=False` when `enable_sleep`; thread `loss` into `_step_currency`
  and call `self.settled = self.sleep_detector.update(loss, self.step_idx)` every
  step; in `_rewire_currency`, when `enable_sleep and self.settled`, prune with
  `sleep_prune_floor`/`sleep_max_prune`, skip grow, append a `{"type":"sleep"}`
  event, and `self.sleep_detector.reset()`.
- [ ] **Step 4** — run, expect PASS; run full `tests/test_train.py`.
- [ ] **Step 5** — commit.

### Task 4: eval `sleep` variant

**Files:** Modify `evals/spec.py` (`VARIANTS`); Test `tests/test_eval_spec.py`.

- [ ] **Step 1 — failing test** (`tests/test_eval_spec.py`):
```python
def test_sleep_variant_is_currency_plus_sleep():
    base = make_config("currency")
    s = make_config("sleep")
    assert base.enable_sleep is False
    assert s.enable_sleep is True
    assert s.grad_currency and s.enable_prune and s.enable_grow
    assert s.confidence_mode == base.confidence_mode      # only sleep differs
    assert s.grow_bar_frac == base.grow_bar_frac
```
- [ ] **Step 2** — run, expect KeyError.
- [ ] **Step 3** — add `"sleep"` to `VARIANTS` (currency config + `enable_sleep=True`).
- [ ] **Step 4** — run, expect PASS.
- [ ] **Step 5** — commit.

### Task 5: full suite + validate.py guardrail

- [ ] Run `.venv/bin/python -m pytest -q` — all pass (incl. new tests).
- [ ] Run `.venv/bin/python validate.py` — must stay 7/7 (sleep off by default).
- [ ] Commit any fixes.

### Task 6: eval + publish

- [ ] `evaluate.py --variants sleep,currency --baseline currency --seeds 5
  --dataset spirals --steps 15000 --jobs 6 --no-cache --publish --run-name
  sleep-consolidation`
- [ ] Commit + push `docs/eval-runs/sleep-consolidation` (pre-authorized).
- [ ] Summarise per the running-sprout-evals protocol (key-metrics table + What
  it means column + honest wins/losses + the README path).
