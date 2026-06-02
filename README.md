# SPROUT — a legible, self-wiring classifier

A tiny feedforward classifier whose brain-inspired mechanics are *directly
observable*: synapses accumulate **confidence**, slow their own learning, get
**pruned**, and new ones **grow** — all on a standard backprop core, on a
sparse graph you can watch rewire itself.

The project has two architectures, both live and both tested:

1. **Gradient-as-currency** *(current default)* — one metered signal drives
   everything. Backprop already computes a per-synapse gradient
   `g_ij = δ_j·a_i` ("how hard, and which way, the loss wants this wire to
   change"). We meter that once into one shared per-wire state — **load**
   `|w|/w̄` and **demand** `M/M̄` — and read it through three lenses:
   **confidence** (freeze a wire that is *important and settled*), **pruning**
   (delete a wire weak in *both* load and demand), **growth** (add the missing
   wire the loss most wishes existed, when it clears a *selective* bar —
   RigL-style). See [sprout/currency.py](sprout/currency.py).

2. **Legacy v1 — eligibility / three-factor** — the original spec
   ([docs/v1_implementation.MD](docs/v1_implementation.MD)): a Hebbian
   "fired-together-recently" eligibility trace gates a global low-error signal to
   drive confidence; growth chases under-firing neurons; pruning uses `|w|·r`.
   Kept verbatim under the `legacy-*` presets — it is more *biologically* flavoured
   and a lateral move on accuracy, with cleaner churn on spirals (see results below).

## Headline result

Both pass **7/7** of the "it works" criteria on two interleaving spirals
(`python validate.py` for currency, `--legacy` for v1). The currency system hits
**99% accuracy** *with no `theta_prune`, `prune_warmup`, or `grow_budget`
tuning* — those three hand-set knobs from v1 are replaced by the one
gradient-aware signal.

| step 0 (all plastic, garbage boundary) | converged (consolidated, fits spirals) |
|---|---|
| ![step 0](docs/assets/spirals_step0.png) | ![final](docs/assets/spirals_final.png) |
| acc ≈ 0.5, every synapse blue (c=0) | acc ≈ 1.0, working pathways red (frozen) |

## Honest comparison: where currency stands

Currency is the **default architecture and the baseline** other variants are
measured against. Three honest truths, all multi-seed (full scorecards under
[docs/eval-runs/](docs/eval-runs/)):

**Accuracy vs legacy is a lateral move, not an upgrade.** Currency matches
legacy's ~0.97–0.99 on spirals from a *single* signal and deletes three tuned
knobs (`theta_prune`, `prune_warmup`, `grow_budget`) — elegance, not a higher
ceiling. The dead-ReLU growth churn that forced `grow_budget` is gone *for free*:
a dead neuron has zero gradient, so candidate wires into it score ~0 and are
never grown. Re-run with
`python evaluate.py --variants legacy-full,currency --baseline currency --seeds 5 --shift 6000`.

**Confidence calibration is a genuine, measured win.** The original tug-of-war
rule *anti-correlated* with real wire utility (`conf_utility_corr ≈ −0.17`): it
froze wires by settledness alone, so it froze *freeloaders*. Re-deriving
confidence as the **2D (importance × settledness)** rule on the *same*
`(load, demand)` state prune reads — with a **softened sigmoid cliff** so a
contested load-bearer keeps some consolidation instead of collapsing to zero —
fixed the sign and made it a significant **+0.31** (15 seeds, no-shift), at **no
accuracy cost** and **zero frozen freeloaders**. Measure calibration on a
*no-shift* run: a mid-run label swap makes the slow confidence EMA lag the
instantaneous post-swap demand, which understates the correlation. See
[docs/eval-runs/2dsoft-vs-2dconf-noshift-15seeds/](docs/eval-runs/2dsoft-vs-2dconf-noshift-15seeds/).

**Grow↔prune oscillation is now largely tamed by a selective grow bar.** Removing
v1's `grow_budget` cap exposed thrash — a wire grown (high virtual gradient),
pruned before it matures, then re-requested because the same virtual gradient is
still there. The fix turned out to be *growth selectivity*, not prune patience:
raising the grow bar to `grow_bar_frac=3.0` (the new default — grow only a wire
the loss wants ≫ a typical live wire) cut the oscillating *population*
(`oscillation_frac` 0.37 → 0.28 ▲), the worst re-grow (`max_regrow` 11 → 6.5 ▲)
and overall churn, and ended ~20% **sparser** (125 → 99 wires) at no accuracy cost
with calibration held. A 15-seed sweep isolates it:
[docs/eval-runs/b1-growbar-sweep/](docs/eval-runs/b1-growbar-sweep/). Two honest
residuals: (1) ~28% of grown wires are still tried twice — the bar lowers thrash
*incidence* but doesn't zero it; (2) damping growth nudges post-shift
`recovered_test_acc` down a little (not significant at 15 seeds, but consistent —
[oscillation-shift-guardrail](docs/eval-runs/oscillation-shift-guardrail/)). An
optional second lever, the **ghost-gradient meter** (`ghost_meter=True` — grow on
a *sustained* EMA so a just-cut wire must re-earn its place), cuts the worst
re-grow further (to ~4) but proved partly redundant once the bar is high
([gb3-ghost-combo](docs/eval-runs/gb3-ghost-combo/)).

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy matplotlib pytest pillow

pytest -q                                   # 189 unit + integration tests

python run.py --preset currency --dataset spirals --steps 15000 --density 0.4
python validate.py                          # currency, all 7 criteria + plots
python validate.py --legacy                 # the v1 eligibility system instead

python evaluate.py --variants currency,legacy-full --seeds 5 --shift 6000
                                            # multi-seed comparative scorecard
```

Artifacts land in `output/<preset>_<dataset>/` (`animation.gif`, frames,
`metrics.json`); the evaluation harness writes its scorecard + plots to
`output/eval/<dataset>_<timestamp>/`.

## Presets (`run.py --preset`)

| Preset | Architecture | Notes |
|---|---|---|
| `core` | plain sparse backprop | all mechanisms off |
| `currency-conf` | currency: + confidence | edges auto-coloured by gradient **demand** |
| **`currency`** *(default)* | currency: confidence + prune + grow | the current architecture (2D calibrated confidence + softened cliff + selective grow bar) |
| `legacy-step1…step5` | v1 build-order, one mechanism at a time | the most legible way to watch each part |
| `legacy-step6` | + homeostasis | opt-in; unstable with ReLU (see deviations) |
| `legacy-full` | full v1 eligibility system | the tuned baseline in the table above |

## What you can watch

The main panel draws neurons as dots (brightness ∝ activation) and synapses as
lines (**thickness ∝ |weight|**). Edge **colour** depends on the mode:

- `confidence` (default): blue = unsure/fast → red = confident/frozen.
- `demand` (currency): dark = settled → bright = the loss is still pushing it.
- `eligibility` (legacy): dark = quiet → bright = co-active "glow".

A line appearing = growth; vanishing = a prune. Side panels: accuracy, synapse
count, and the 2-D decision boundary. `validate.py` also writes `eff_lr.png`
(confidence ↑ ⇒ effective LR ↓), `selectivity.png` (the metered signal is
selective), and `decay.png` (confidence falls after a concept shift).

## How the currency works (formulas)

Two EMAs per wire are the whole currency:

```
M_ij ← β·M_ij + (1−β)·|g_ij|     # magnitude meter — "how hard am I pushed"
S_ij ← β·S_ij + (1−β)· g_ij      # signed meter    — "which way, on net"
load        ℓ = |w_ij| / mean(|w|)        # weight vs the network (carries load now?)
demand      d = M_ij / mean(M)            # gradient vs the network (still wanted?)
consistency κ = |S_ij| / (M_ij + ε)       # tug-of-war variant: 1 = same dir, 0 = contested
```

`load` and `demand` are the shared 2D state; confidence and pruning are two
lenses on it (one source of truth, [sprout/currency.py](sprout/currency.py)'s
`network_scales`/`load`/`demand`).

- **Confidence / plasticity** (`update_confidence_2d`, default): freeze a wire
  only when it is **important *and* settled**. `imp = (ℓ−1)₊` (above-average
  load), `settled = σ(k·(1−d))` (a *softened* cliff — demand below average ⇒
  settled, smoothly), `c ← EMA toward gain·imp·settled` clipped to `[0, c_max]`;
  effective LR is `η/(1+c)`. Reading the *same* `(load, demand)` state prune
  reads is what makes confidence *track* utility instead of fighting it, and the
  hard `imp` floor means freeloaders (below-average load) never freeze. The prior
  tug-of-war rule (`update_confidence_currency`, `confidence_mode="tugofwar"`)
  earned from `κ·(1−d)₊` and lost from `(d−1)₊` — kept as a variant for comparison.
- **Pruning** (`prune_currency`): utility `|w|/w̄ + λ·M/M̄`; cut only wires weak in
  **both** senses. Protects small-but-wanted newborns ⇒ no warmup needed.
- **Growth** (`batch_edge_scores` + `grow_currency`): score missing wires by
  their *virtual* gradient `δ_j·a_i`; grow only those wanted **far more than a
  typical live wire** (`grow_bar_frac=3.0` — a *selective* hiring bar that keeps
  rewiring calm and sparse and tames the grow↔prune oscillation), born at weight
  0. Dead neurons (`δ_j=0`) score ~0 and are never grown. Optional:
  `ghost_meter=True` grows on a *sustained* EMA of the virtual gradient so a
  just-cut wire must re-earn its place (`update_ghost_meter`).

The full design, including the honest trade-off discussion, is the basis for
[sprout/currency.py](sprout/currency.py)'s module docstring.

## Code layout

```
sprout/
  data.py        generate_blobs / generate_spirals
  network.py     Neuron, Synapse (+ grad_mag/grad_signed meters); forward/backward
  learning.py    legacy: firing-rate, eligibility, three-factor confidence, gated update
  currency.py    current: gradient meters + confidence/prune/grow readouts
  plasticity.py  legacy: prune (|w|·r), grow (activity), homeostasis
  viz.py         render_frame (confidence / demand / eligibility edges) + make_gif
  train.py       Config (both stacks behind flags) + Trainer (grad_currency branch)
run.py           experiment driver / CLI (currency default, legacy-* presets)
validate.py      validation harness (currency default; --legacy for v1)
evaluate.py      comparative eval entry: multi-seed scorecard + diagnostic plots
evals/           eval harness package (spec, runner, metrics, aggregate, report, cli)
tests/           TDD suite (data, network, learning, plasticity, train, currency, infra, eval)
```

Pure NumPy; forward/backward are hand-rolled over adjacency lists so the
irregular, mutating sparse graph is handled directly (no dense layers).

## Deviations & known limitations

**Trade-off of the currency architecture:** it replaces the Hebbian eligibility
trace (the most *biologically local* part of v1) with the backprop gradient. The
result is more functional and unified but openly "backprop, read three ways" —
less biologically plausible. For a project about *legibility*, leading with the
clearer single-cause story is the deliberate choice; the local Hebbian version
remains one command away (`--legacy`).

**Legacy-v1 deviations** (discovered empirically; each documented in-code):
eligibility clamped ≥0; confidence reads eligibility as a *bounded gate* not a
raw multiplier (unbounded froze half-trained synapses); homeostasis off by
default (ReLU + weight-rescaling diverges); `grow_budget` to stop dead-ReLU
growth churn; `theta_prune`/`prune_warmup` tuned so pruning doesn't sever
mid-training wires; network `[2,16,16,16,2]` and spirals `turns=1.0, noise=0.10`.

**Default topology + horizon.** The default hidden layers were promoted from
`10,10,8` to a uniform **16** (`[2,16,16,16,2]`), and the single-task training
horizon shortened to **15k steps**. The [neuron-width sweep](docs/eval-runs/neuron-width-sweep/)
found w16 the accuracy/speed sweet spot — near-top accuracy, ~1.8× faster
convergence than the old net, and the fewest idle units — and w16 converges
comfortably inside 15k. (`validate.py` stays pinned to the original net/horizon
as a fixed regression guardrail.)

**The one genuinely unsolved problem (both architectures): reviving dead ReLU
units.** A neuron whose pre-activation is always negative emits zero gradient, so
*no* growth rule — activity-based or gradient-based — can revive it (a wire into
it gets no learning signal). Currency handles this gracefully (never wastes
growth there); it does not *solve* it. Fixes on the list: a small non-zero birth
weight, or a bias nudge for chronically-dead units.

## Next steps

Confidence **calibration is resolved** (the 2D + softened-cliff redesign) and
grow↔prune **oscillation is largely tamed** by the selective grow bar (the
default; see "Honest comparison" above). Remaining:

1. **Oscillation residuals** — the selective bar lowers thrash *incidence* but
   ~28% of grown wires are still tried twice, and damping growth nudges post-shift
   recovery down slightly. The opt-in ghost meter (A2) cuts intensity further; a
   cleaner *recovery-preserving* lever is still open.
2. **Dead-unit revival** — small non-zero birth weight or bias nudge.
3. **Stable homeostasis** — a per-neuron trained gain instead of multiplicative
   weight rescaling.
4. Parked v2 ideas: spiking neurons + surrogate-gradient STDP, recurrence,
   confidence-gated exploration noise, a sleep/replay consolidation phase.
