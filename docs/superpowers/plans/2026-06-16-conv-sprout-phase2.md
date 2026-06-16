# Conv-SPROUT Phase 2 — Implementation Plan

> **For agentic workers:** build in stages, TDD, commit per stage. The numerical
> gradient check in Stage A is the linchpin — nothing trains until it passes.

**Goal:** A weight-shared conv layer governed by gradient-as-currency that learns
and self-sizes its filters, feeding the existing sparse phasic head — additive code
only. See `docs/superpowers/specs/2026-06-16-conv-sprout-phase2-design.md`.

**Architecture:** image → ConvEconomy(K filters) → ReLU → 2×2 maxpool → flatten →
existing head Network. Simultaneous wake; phasic rewire of both at a shared
settledness plateau.

---

### Stage A — `sprout/conv.py` pure conv/pool math (TDD; gradient check is the gate)

**Files:** Create `sprout/conv.py`, `tests/test_conv.py`.

Functions: `conv_valid_forward(img, kernels)→preact (K,oh,ow)`;
`conv_valid_filter_grad(img, d_preact)→g (K,kh,kw)`; `maxpool_forward(a,p)→(pooled,argmax)`;
`maxpool_backward(d_pooled, argmax, shape)→d_a`; `filter_bank` reuse from
`conv_features` for the fixed arm.

Tests:
- forward matches a naive triple-loop conv on a small input.
- maxpool forward picks block maxima; backward routes gradient only to argmax.
- **NUMERICAL GRADIENT CHECK:** for a tiny pipeline `loss = Σ relu(conv)` (and a
  pooled variant), analytic `conv_valid_filter_grad` matches finite differences on
  each kernel tap to `< 1e-5`.

Commit: `feat: conv/pool forward+backward (gradient-checked)`.

---

### Stage B — `ConvEconomy` (the filter-level currency) (TDD)

**Files:** extend `sprout/conv.py`, `tests/test_conv.py`.

`ConvEconomy(K, kh, kw, seed, cfg-ish knobs)` state: `theta (K,kh,kw)`, per-filter
`M, Svec (K,kh,kw), conf (K,), age (K,)`. Methods:
- `forward(img)→(feat_vector, cache)` (conv→relu→pool→flatten; cache for backward).
- `backward(d_feat, cache)→g (K,kh,kw)` (unflatten→unpool→relu-grad→filter grad).
- `update_meters(g, beta_g)`; `update_confidence(gain, alpha, c_max, k)` (reuse
  `currency.settledness`; `imp=max(load−1,0)`); `gated_update(g, eta)` (`η/(1+c)` per filter).
- `prune(floor, lam, grace, k_min)→pruned ids`; `grow(mode, k_max)→born ids`
  (split highest-demand or random); `load()`, `demand()` vectors.

Tests: forward feat length = K·poh·pow; meters EMA correctly; confident filter
(high load,low demand) gets high conf → small step; newborn (low load,high demand)
protected from prune; prune drops the lowest-utility filter, keeps ≥k_min; grow
respects k_max and split clones the highest-demand filter.

Commit: `feat: ConvEconomy — per-filter gradient-as-currency`.

---

### Stage C — `ConvModel` + `ConvTrainer` joint wake (TDD)

**Files:** Create `sprout/conv_train.py`, `tests/test_conv_train.py`.

`ConvModel(conv, head_net)`. `ConvTrainer(cfg, model, X_imgs, y, seed)`:
- `step()`: sample; conv.forward → head forward (`net.forward(feat)`); head backward
  (`net.backward`); compute **input-layer delta** `d_feat[p]=Σ_post w·delta[post]`;
  conv.backward(d_feat); update head (reuse `currency.update_gradient_meters`,
  `currency.update_confidence_2d`, `learning.apply_gated_update`) and conv
  (update_meters/confidence/gated_update); feed joint loss to a `SettlednessDetector`.
- `predict(X_imgs)` / accuracy helper.

Tests: model forward shapes; `ConvTrainer` reaches > chance on a tiny synthetic
image task in a few k steps; deterministic for a fixed seed.

Commit: `feat: ConvModel + ConvTrainer joint wake step`.

---

### Stage D — phasic rewire (head + conv) (TDD)

**Files:** extend `sprout/conv_train.py`, `tests/test_conv_train.py`.

At a settledness plateau (`SettlednessDetector.update` True, past warmup): snapshot
a batch of current conv features; head rewire via `currency.batch_edge_scores` +
`grow_currency` + `prune_currency` (the real functions); conv rewire via
`ConvEconomy.prune`/`grow`; then `detector.reset()`. Log events
(`{"type":"conv_grow"|"conv_prune"|"sleep", ...}`).

Tests: a sleep event fires on a settled tiny run; filter count changes within
[k_min,k_max]; head synapse count changes; determinism holds.

Commit: `feat: phasic rewire for Conv-SPROUT (head reuse + filter grow/prune)`.

---

### Stage E — experiment runner + E1

**Files:** Create `conv_experiment.py`.

Multi-seed train; per-arm mean±std test acc over time; writes
`docs/eval-runs/<name>/`: README (metrics table + seed-bootstrap ▲/≈ vs the
fixed-hand baseline), `acc_curves.png`, `filters_<arm>.png` (kernels imshow),
`metrics.json`. Arms: `fixed-hand`, `learned-gated` (K=6, K=12). 14×14 MNIST, ≥3
seeds.

Run + commit + push the folder.

---

### Stage F — E2 (self-sizing) + summary

Enable phasic filter grow/prune; start K0, cap K_max; report final-K trajectory +
filter viz. Run, publish, push. Then summarise findings (wins AND losses) + the
Phase-2 verdict; update memory.
