# SPROUT — a legible, self-wiring classifier

A tiny feedforward classifier whose brain-inspired mechanics are *directly
observable*: synapses accumulate **confidence**, slow their own learning, get
**pruned**, and new ones **grow** — all on a standard backprop core, on a sparse
graph you can watch rewire itself.

## The architecture: gradient-as-currency

One metered signal drives everything. Backprop already computes a per-synapse
gradient `g_ij = δ_j·a_i` ("how hard, and which way, the loss wants this wire to
change"). We meter it once into one shared per-wire state — **load** `|w|/w̄` and
**demand** `M/M̄` — and read it through three lenses:

- **confidence** — freeze a wire that is *important and settled* (2D rule;
  effective learning rate `η/(1+c)`).
- **pruning** — delete a wire weak in *both* load and demand.
- **growth** — add the missing wire the loss most wants, when it clears a
  *selective* bar (RigL-style). The default `grow_demand_k=4` prices only the
  top-4 highest-demand post neurons, so the candidate scan scales like
  `k·active_pre` rather than all active pre/post pairs.

Structural change is **phasic**, on one smoothed-loss state machine: the net
**learns awake** (gated-SGD + metering, no rewiring), **rewires asleep** (one
prune-the-weak + grow-the-wanted pass, fired only once the loss has *settled*
onto a plateau), and **hires when startled** (a grow-only pass ~60 steps into a
sustained loss *spike* — a real regime change). The default architecture is
**`phasic-startle-k4`**. See [sprout/currency.py](sprout/currency.py),
[sprout/sleep.py](sprout/sleep.py), [sprout/train.py](sprout/train.py).

## Tasks & datasets

One registry serves them all ([sprout/datasets.py](sprout/datasets.py),
`get_dataset`):

| name | what | use |
|---|---|---|
| `spirals` | two interleaving spirals (2-D, 2-class) | the home task + `validate.py` guardrail |
| `blobs` | two Gaussian blobs (2-D) | pipeline debug |
| `digits` | scikit-learn 8×8 digits (64-D, 10-class) | small real multiclass |
| **`mnist`** | **MNIST 2×2-pooled to 14×14 (196-D, 10-class) — the default MNIST** | the main image benchmark |
| `mnist-full` | full 28×28 MNIST (784-D) | full-resolution control |

**Why 14×14 is the default MNIST:** at a fixed sparse edge budget, a 14×14
thumbnail the net can *fully cover* beats a 784-pixel image it can only glimpse —
2×2 pooling discards little that matters for digits, and full resolution
underperformed at every budget tested (see findings). `mnist-full` stays for that
comparison.

## Headline result (spirals)

The currency system passes **7/7** "it works" criteria on two interleaving
spirals (`python validate.py`), hitting **99% accuracy** with *no* `theta_prune`,
`prune_warmup`, or `grow_budget` tuning — those hand-set knobs are replaced by the
one gradient-aware signal.

| step 0 (all plastic, garbage boundary) | converged (consolidated, fits spirals) |
|---|---|
| ![step 0](docs/assets/spirals_step0.png) | ![final](docs/assets/spirals_final.png) |
| acc ≈ 0.5, every synapse blue (c=0) | acc ≈ 1.0, working pathways red (frozen) |

## What we found (multi-seed; full scorecards in [docs/eval-runs/](docs/eval-runs/))

**Architecture (spirals).** Currency is a *lateral* accuracy move from a single
signal that deleted three tuned knobs. The **2D (importance × settledness)**
confidence rule fixed calibration (`conf_utility_corr` −0.17 → **+0.31**: it now
tracks real wire utility instead of freezing freeloaders). **Phasic** structure
ends **~47% sparser** at preserved accuracy and makes grow↔prune oscillation
structurally impossible (rewires fire only at far-apart plateaus). **Startle**
adds demand-triggered hiring that is inert until the world changes (0 false alarms
on stationary data). A legacy Hebbian-eligibility stack was removed once currency
proved a lateral move from one signal; its design notes live in
[docs/v1_implementation.MD](docs/v1_implementation.MD).

**Sparse vs dense at constant compute (digits → MNIST)** —
[full write-up](docs/findings-2026-06-14-sparse-vs-dense-constant-compute.md):

- On an *easy* task (8×8 digits) a small **dense** net wins; wide-sparse only ties.
- On a *harder* task (**MNIST 14×14**) a **wide self-rewiring *sparse*** net
  **beats** small-dense at the same edge budget (+4.8pt), and the gap **grows with
  data** (+14.8pt at 4× data) because the small dense net underfits.
- **`mnist-w32-sparse`** (one hidden layer, fan-in ~98) is the **robust default**.
- The ~0.93 MNIST ceiling is **architectural**: width *plateaus*, **depth hurts**,
  **higher resolution hurts** at fixed budget, **more budget is a dead end** (the
  net self-prunes to ~4–5k edges regardless) — only *more data* reliably helps.
  Breaking it needs a different prior (convolution / weight-sharing), not more MLP.

> **Rule of thumb:** at a fixed compute budget, spend it on a *wider, self-rewiring
> sparse* net whenever the task is hard enough that a small dense net underfits —
> and that edge grows with data.

## Conv-SPROUT — the economy grows its own convolutions (Phase 2)

Breaking the ~0.93 MNIST ceiling needed a **convolution / weight-sharing** prior —
so Conv-SPROUT adds one *inside the same gradient-as-currency economy*, rather than
bolting on a black box. The economic unit moves up one level, from the **wire** to
the **filter**: a filter's taps are its weight-shared parameters, applying it
densely across the image is its placements, and a single gradient meter on the
filter aggregates the gradient over every placement (= the conv weight gradient).
The same readouts run one level up — 2D confidence, demand-driven pruning,
plateau-gated growth, `η/(1+c)` gated learning. A learned conv front-end feeds the
**existing `phasic-startle-k4` head** unchanged ([sprout/conv.py](sprout/conv.py),
[sprout/conv_train.py](sprout/conv_train.py)).

The **promoted baseline recipe** (the `conv-sprout` arm; defaults in `ConvTrainer`):

1. **Learn** the filters (don't hand-design them).
2. **Consolidate** — wind the filter learning rate down over training (cosine), so
   the filters settle and the head finishes on a *stationary* representation. This
   is the load-bearing fix: without it learned filters drift late and *lose* to a
   fixed bank; with it they *win*.
3. **Self-size** — start generous (12 filters) and **trim functional redundancy**
   (prune a filter whose pooled outputs correlate `> 0.80` with a more-useful one)
   down to a lean bank (~7).

**Result (MNIST 14×14, 60k steps, 5 seeds —
[findings](docs/findings-2026-06-16-conv-sprout-phase2.md)):** Conv-SPROUT reaches
**~0.95**, clearing the old ~0.93 MLP ceiling and beating the fixed hand-filter bank
(0.931); at *matched* filter count (~7) the self-sized learned bank still wins
(**0.951 vs 0.931**). On a filter-sensitive control task (matched-filter motifs) the
gain is **+25–33pt**. Honest caveats: the win needs a real budget (≤15k steps still
favors the cheap fixed bank), and gradient-trained filters diversify naturally, so
redundancy trimming is real-but-bounded (~12 → ~7 free; lower cutoffs trade
accuracy). Run it: `python conv_experiment.py` (defaults to `fixed-hand-k6` vs
`conv-sprout`).

## Vectorized backend (the fast path)

The per-synapse Python loop costs `O(edges)`/step. [sprout/fast.py](sprout/fast.py)'s
`ArrayNet` runs the *same* network as flat NumPy edge-arrays
(scatter/segment-sums), **parity-tested against the object model** and **~60×
faster at scale** (a ~94k-edge net: 177 → 2.8 ms/step). The whole eval harness
runs on it via `--backend array`. The object model stays the **default and
validated reference**; array results are statistically equivalent but not
identical (float drift), so A/B *within* a backend. Design:
`docs/superpowers/specs/2026-06-15-vectorized-backend-design.md`.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy matplotlib pytest pillow scikit-learn   # scikit-learn: digits + MNIST
pytest -q                                                 # unit + integration tests

python validate.py                                        # spirals, 7/7 criteria + plots
python run_dataset.py --dataset mnist --steps 40000 --layers 196,64,10   # quick single run
python evaluate.py --variants phasic-startle-k4,fully-connected --baseline fully-connected \
    --dataset mnist --backend array --seeds 5 --steps 30000 --points 12000   # scorecard
```

Single runs write to `output/...`; the eval harness writes a scorecard + plots to
`output/eval/<dataset>_<ts>/`, and `--publish` packages a git-committable,
phone-readable run folder under `docs/eval-runs/`.

## Entry points

- **`validate.py`** — the spirals 7/7 "it works" guardrail (+ diagnostic plots).
- **`run_dataset.py`** — one quick training run on any registry dataset (fast,
  no harness); defaults to `mnist`.
- **`run.py --preset {core, currency-conf, currency}`** — the spirals viz driver
  with the animated GIF (`currency` = the phasic-startle-k4 model).
- **`evaluate.py --variants ... [--dataset ...] [--backend object|array]`** — the
  multi-seed comparative harness. Key variants: `phasic-startle-k4` (default),
  `fully-connected` (dense control), and the constant-compute sweep arms
  `digits-*` / `mnist-*` (14×14) / `mnist784-*` (full). See [evals/spec.py](evals/spec.py).

## How the currency works (formulas)

Two EMAs per wire are the whole currency:

```
M_ij ← β·M_ij + (1−β)·|g_ij|     # magnitude meter — "how hard am I pushed"
S_ij ← β·S_ij + (1−β)· g_ij      # signed meter    — "which way, on net"
load   ℓ = |w_ij| / mean(|w|)    # weight vs the network (carries load now?)
demand d = M_ij  / mean(M)       # gradient vs the network (still wanted?)
```

`load` and `demand` are the shared 2D state; confidence and pruning are two lenses
on it (one source of truth in [sprout/currency.py](sprout/currency.py)).

- **Confidence** (`update_confidence_2d`): freeze a wire only when *important and
  settled* — `imp=(ℓ−1)₊`, `settled=σ(k·(1−d))`, `c ← EMA toward gain·imp·settled`,
  clipped to `[0,c_max]`. The hard `imp` floor means below-average-load freeloaders
  never freeze.
- **Pruning** (`prune_currency`): utility `|w|/w̄ + λ·M/M̄`; cut only wires weak in
  **both** senses (protects small-but-wanted newborns — no warmup needed).
- **Growth** (`batch_edge_scores` + `grow_currency`): score missing wires by their
  *virtual* gradient `δ_j·a_i`; grow only those wanted ≫ a typical live wire
  (`grow_bar_frac=3.0`), born at weight 0. Dead neurons (`δ_j=0`) score ~0 and are
  never grown.

## Code layout

```
sprout/
  data.py        generate_blobs / generate_spirals (2-D tasks)
  datasets.py    get_dataset registry: spirals/blobs/digits/mnist/mnist-full
  network.py     Neuron, Synapse (+ gradient meters); object forward/backward
  fast.py        ArrayNet / ArrayTrainer — vectorized backend (--backend array)
  learning.py    firing-rate EMA + confidence-gated weight update
  currency.py    gradient meters + confidence/prune/grow readouts
  sleep.py       SettlednessDetector — the loss-plateau gate / phasic trigger
  conv_features.py fixed filter bank + conv/pool (Phase-1 preprocessing front-end)
  conv.py        Conv-SPROUT: gradient-checked conv/pool + ConvEconomy (filter-level currency)
  conv_train.py  ConvModel + ConvTrainer — joint conv+head training (Phase 2)
  viz.py         render_frame (confidence / demand edges) + make_gif
  train.py       Config + Trainer (phasic vs continuous structural plasticity)
run.py           spirals viz driver / CLI (currency presets)
run_dataset.py   quick single-run smoke test on any registry dataset
conv_experiment.py  Conv-SPROUT multi-seed runner (default: fixed-hand-k6 vs conv-sprout)
validate.py      the 7 "it works" criteria on spirals (+ plots)
evaluate.py      comparative eval: multi-seed scorecard + diagnostic plots
evals/           eval harness package (spec, runner, metrics, aggregate, report, cli)
tests/           TDD suite (data, datasets, network, fast, learning, train, currency, sleep, eval)
```

Pure NumPy. The object path hand-rolls forward/backward over adjacency lists so
the irregular, mutating sparse graph is handled directly; the `fast` backend runs
the same graph as parallel edge arrays.

## Research roadmap — performance levers

Ranked ideas for improving **prediction accuracy** while staying faithful to the
gradient-as-currency architecture (sparse, self-wiring, phasic, legible). Guiding
principle: the most faithful gains **reuse or extend the currency** rather than
bolt on something foreign.

| Lever | Idea | Why high-potential | Faithfulness | Status |
|---|---|---|---|---|
| **Currency-native optimizer** | step along `S/(M+ε)` — the two meters *become* the optimizer: a self-normalizing, auto-annealing per-wire adaptive step | adaptive per-wire LR is one of ML's most reliable accuracy/convergence wins; **zero new state**; lifts every task | **maximal** (reuses the currency) | **in progress** — `Config.optimizer="currency"`, `opt-currency-*` variants |
| **Data augmentation** | random small image shifts / jitter | historically the single biggest MNIST lever; synergistic with conv translation-invariance | high (pure data; economy untouched) | planned |
| **Mini-batch gradients** | average `g` over 16–32 samples instead of batch=1 | less noisy gradients → better convergence **and** cleaner `M`/demand → smarter prune/grow | high (only the gradient estimate changes) | planned |
| **Stacked / deep conv** | a 2nd `ConvEconomy` on top of the first | compositional features (edges→strokes→parts) — the big swing on images | high (same currency governs both) | planned |
| **Neuron-level growth** | birth/prune whole *neurons* where demand concentrates, not just edges | raises the capacity ceiling on hard tasks without manual width tuning | high (natural currency extension) | planned |

Lower-tier (cheap, modest): head LR warmup/cosine schedule; label smoothing /
output temperature.

## Limitations & next steps

- **The MNIST ceiling was architectural, and Conv-SPROUT addresses it.** Every MLP
  lever (width/depth/resolution/budget) is mapped and exhausted; the missing prior
  was **convolution / weight-sharing**, now added *inside the economy* (see
  Conv-SPROUT above) — it clears ~0.93 → **~0.95**. Open conv items: deeper
  (stacked) conv layers, and a redundancy-trim that can lean below ~7 filters
  without an accuracy cost.
- **Dead ReLU units** are an absorbing state: a unit whose pre-activation is always
  negative emits zero gradient, so no growth rule can revive it. Currency handles
  it gracefully (never wastes growth there) but does not *solve* it; phasic
  structure roughly doubles permanently-dead units. Candidate fixes: small non-zero
  birth weight, or a bias nudge for chronically-dead units.
- **Continual / forgetting regime** — phasic's sparsity buys a small second-task
  acquisition cost vs the continuous baseline; the dead-unit interaction there is
  the main open measurement.
- **Stable homeostasis** — a per-neuron trained gain (intrinsic plasticity) instead
  of multiplicative weight rescaling, to make activation sparsity real.
- Parked v2 ideas: spiking neurons + surrogate-gradient STDP, recurrence,
  confidence-gated exploration noise.
```
