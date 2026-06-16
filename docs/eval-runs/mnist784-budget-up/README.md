# MNIST 784 budget-up sweep — does denser input coverage beat 14×14?

- **Date:** 2026-06-15 · **Branch:** sprout-v1
- **Run via:** `sprout.fast.train_array` (the new vectorized backend) — *not* the
  eval harness (the `--backend` wiring is phase 2). 12 runs (~5 min total) that
  would have taken hours on the object Trainer; this experiment is what the
  backend was built to make tractable.
- **Setup:** full 784-pixel MNIST, hidden width 32 `(784, 32, 10)`, phasic-startle-k4
  config, 12k train / 1k test, 60k steps, 3 seeds. The **edge budget is swept up**
  by raising density (= raising fan-in *coverage* of the 784 input, the lever that
  was starved in the earlier `mnist784-depth-1v2` run).

## The question

Earlier (`mnist784-depth-1v2`) full 784 at ~6k edges scored 0.914 — *below* 14×14's
0.929 — because each neuron's fan-in covered only ~10–25% of the input. Now that the
backend makes big budgets affordable: **does densely covering the 784 input
(fan-in 25% → 100%) beat the downsampled 14×14?**

## Results (all via `train_array`, same code path → fair comparison)

| input | coverage | fan-in | edges start→end | test acc (3 seeds) |
|---|---|---|---|---|
| **14×14 ref** | 50% | 98 | 3.3k → ~4k | **0.926 ± 0.003** |
| 14×14 ref (w64) | 25% | 49 | 3.3k → ~4k | 0.926 ± 0.006 |
| 784 | 25% | 196 | 6.4k → 3.6k | 0.899 ± 0.001 |
| 784 | 50% | 392 | 12.7k → 4.4k | 0.909 ± 0.005 |
| 784 | **75%** | 588 | 19k → 4.8k | **0.921 ± 0.006** (best 784) |
| 784 | 100% | 784 | 25k → 5.0k | 0.903 ± 0.006 |

## Verdict: NO — denser coverage does not beat downsampling

1. **Coverage helps, then hurts.** Full-784 accuracy rises 0.899 → 0.921 as coverage
   goes 25% → 75%, then **drops to 0.903 at 100%** (dense first layer). Non-monotonic;
   the peak is at 75%.
2. **Even the best full-784 (0.921, 19k edges) loses to 14×14 (0.926, 3.3k edges)** —
   on the *same* code path. Full resolution needs **~6× the edges** just to get
   *close*, and never catches up.
3. **The net self-prunes to ~4–5k edges regardless of starting budget** (25k → 5k at
   100%). It "wants" ~4–5k edges no matter how many you hand it; extra starting
   budget mostly gets pruned away, and a dense init's prune-churn actively hurts.

**Why:** 2×2 pooling discards little that matters for digits (adjacent pixels are
highly redundant), so 14×14 is a *better-conditioned* input for a small sparse MLP,
while full-784's extra dimensions are redundant capacity the net can't use
efficiently. **Increasing the budget is a dead end here — diminishing then negative
returns.** This reinforces the architectural ceiling: within the sparse-MLP family,
neither width, depth, resolution, nor budget breaks ~0.93. The remaining lever is a
different prior (convolution / weight-sharing), not more of the same.

## Cross-check / caveat

`train_array`'s 14×14 w32 = 0.926 vs the harness's 0.929 (−0.3pt) and 784 d=0.25 =
0.899 vs harness 0.914 (−1.5pt): expected float-drift trajectory divergence over 60k
steps (startle is inert on single-task data — 0 events — so it is *not* the cause).
The **within-`train_array`** comparison above is apples-to-apples and is the basis
for the verdict.

## Reproduce

```python
from sprout.datasets import get_dataset
from sprout.network import build_graph, init_weights
from sprout.fast import train_array
from sprout.train import Config, accuracy
cfg = Config(eta_base=0.02, enable_confidence=True, enable_prune=True,
             enable_grow=True, gamma_dec=0.001, t_struct=200,
             phasic_structure=True, startle=True, grow_demand_k=4)
Xtr, ytr, Xte, yte = get_dataset("mnist", seed=0, n_points=12000)   # full 784
net = build_graph([784, 32, 10], density=0.75, seed=0); init_weights(net, seed=0)
net = train_array(cfg, net, Xtr, ytr, seed=0, steps=60000)
print(accuracy(net, Xte, yte))   # ~0.92
```
