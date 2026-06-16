# Wide-sparse vs small-dense at constant compute (digits → MNIST14)

**Date:** 2026-06-14 · **Branch:** sprout-v1

## The question

At a **fixed compute budget** (≈ number of live edges; SPROUT's per-step cost is
`O(edges)`), is it better to spend it **densely on a few neurons** or **sparsely
across many**? And does the answer depend on the task?

All "sparse" arms are the promoted self-rewiring architecture
(`phasic-startle-k4`: phasic prune+grow at settledness plateaus + startle); the
"dense" arm is a static fully-connected MLP (plasticity off). Budgets are held
constant by trading width against initial density. Every run is multi-seed and
published under `docs/eval-runs/` (mean ± std, ▲/▼/≈ = 95% bootstrap CI vs the
arm's baseline).

## The arc

| # | Run (`docs/eval-runs/…`) | Setup | Headline |
|---|---|---|---|
| 1 | `digits-sparse-vs-fc` | 8×8 digits, sparse vs full FC | sparse ≈ dense acc on ~⅓ the edges (it's competitive) |
| 2 | `digits-width-sweep-constant-compute` | 8×8, ~1184 edges, w16-dense vs w32/64/128-sparse | **easy task: small-dense wins; wide-sparse only ties** (w32) or loses (w64/128) — fan-in starvation |
| 3 | `digits-w128-kscale` | w128, grow_demand_k 4/16/32 | **`k` is not the fix** — more growth + higher fan-in, but accuracy flat; `neurons×fan-in≈edges` is the wall |
| 4 | `digits-budget-floor` | 8×8, shrink the budget | no free floor: graceful ~1pt/halving; dense-parity needs ~697 edges |
| 5 | `mnist14-width-sweep` | **MNIST 14×14** (196-in, harder), ~3296 edges | **crossover: wide-sparse BEATS dense +4.8pt on ~40% fewer edges** |
| 6 | `mnist14-scale-12k-60k` | 4× data, 3× steps | **gap explodes to +14.8pt** — dense underfits harder with more data (train acc 0.78); w32-sparse 0.929 |
| 7 | `mnist14-scale-24k-120k` | 8× data, 6× steps | **plateau ≈0.93; width gap closes** (w32≈w64≈w128); w32 reaches 90% cheapest |
| 8 | `mnist14-widen-budget` | 2× budget on wide arms | plateau is *soft*: w128-b2 0.938 (+0.9pt) but at ~2× compute — not worth it |

## What we learned

1. **The deciding factor is task headroom, not sparsity per se.**
   - *Easy task* (8×8 digits, a small dense net already saturates ~97%):
     small-dense wins; wide-sparse only matches. Spreading a fixed budget over
     more neurons starves fan-in (w128 ≈ 7 incoming) and there's no headroom for
     the extra neurons to exploit.
   - *Hard task* (MNIST 14×14, a small dense net underfits): **wide-sparse
     wins** — it has the neurons to represent the task, and the self-rewiring
     keeps them cheap.

2. **The advantage grows with data.** As training data increased (3k→12k), the
   small dense net got *worse* (capacity bottleneck: train acc fell to ~0.78),
   while wide-sparse improved — the gap went **+4.8 → +14.8pt**.

3. **The optimal width widens with data, then widths converge.** At little data
   the lean `w32` led the sparse arms; with more data `w64`/`w128` caught up
   until all three tied (~0.93). More data feeds more neurons.

4. **`grow_demand_k` is not a capacity knob.** Scaling it with width spreads
   growth and lifts fan-in a little, but cannot beat the `neurons × fan-in ≈
   edges` arithmetic. Fan-in is set by the budget, not the grow scan.

5. **Diminishing returns past the matched budget.** Doubling edges on the wide
   arms nudged accuracy ~+1pt (0.929 → 0.938) for ~2× compute. The MNIST14
   ceiling (~0.93–0.94, never 95%) is mostly **data/resolution-limited** (14×14
   discards detail), with only a small capacity component.

## The rule (practical takeaway)

> **At a fixed compute budget, spend it on a wider, self-rewiring *sparse*
> network whenever the task is hard enough that a small dense net underfits — and
> that advantage *grows* with data. If the task is easy enough that a small dense
> net already saturates, sparsity only matches it.**
>
> **`w32-sparse` (lean, ~fan-in 98) is the robust default across every regime
> here**: it ties the best on easy digits at low compute, wins on MNIST14, and
> stays the most compute-efficient at scale (same plateau accuracy, fewest
> edge-steps to reach it). Only widen the budget if you specifically need the
> last ~1 point and can pay ~2× for it.

## Caveats

- These are small nets at small scale; MNIST14 tops out ~0.93–0.94 (not SOTA) —
  the comparisons are **relative**, which is exactly the question asked.
- 14×14 downsampling caps the achievable accuracy; full 28×28 MNIST needs a
  vectorized forward/backward first (the per-synapse Python loop is `O(edges)`).
- Scale runs (#6–8) use 3 seeds (the 5-seed #5 already established the core
  win); `train_eval_cap` bounds per-snapshot train-metric cost on large data.
