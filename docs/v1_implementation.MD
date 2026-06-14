# SPROUT — A Toy Architecture for Confidence-Gated Structural Plasticity

**Codename:** SPROUT (Sparse, Plastic, Reorganizing, Online-Updating network). The name is just a handle for the repo.

**Status:** v1 toy spec, for validation of mechanics — *not* a performance benchmark.
**Audience:** implementing engineer. Assumes familiarity with backprop, NumPy/PyTorch, and matplotlib.
**Primary goal:** make a small network whose novel biological mechanics are **directly observable** — you can watch synapses fire, gain confidence, slow their learning, get pruned, and watch new ones grow.

---

## 1. What we are building and why

A small feedforward classifier on 2D points, with a standard backprop core, augmented by three coupled biological mechanisms:

1. **Structural plasticity** — the connection graph is sparse and changes during training (synapses are pruned and grown).
2. **Per-synapse confidence + confidence-gated learning rates** — each synapse tracks how "sure" it is and slows its own learning as it becomes confident (a synaptic-consolidation effect).
3. **Three-factor / eligibility-trace learning** — a local "fired together recently" signal on each synapse, combined with the global backprop error, drives confidence.

The whole design is optimized for **legibility, not power**. The success criterion is *"we can clearly see the mechanics working as intended on a problem where we know the right answer,"* not accuracy on a hard task. Keep the network tiny and the structural changes slow and visible.

---

## 2. Key design decisions (read before implementing)

These are deliberate. Please do **not** "optimize them away" — several look like bugs but are the point of the project.

| Decision | Why it's this way |
|---|---|
| **Rate-based net, not spiking** | In a plain feedforward net there is no within-pass timing, so "presynaptic fires before postsynaptic" is meaningless. We reinterpret "fire together" as **co-activation in the same forward pass**, tracked by an eligibility trace. Spiking is deferred to v2. |
| **Feedforward DAG (layered)** | Keeps backprop simple. Growth is only allowed from an earlier layer to a later layer so the graph stays acyclic. Recurrence is v2. |
| **Prune by *utility*, not confidence** | Low confidence means "still actively learning" (high learning rate). Pruning low-confidence synapses would cull exactly the ones mid-training. We prune by utility = `|weight| × presynaptic activity`. |
| **Confidence can *decrease* (slow decay)** | Both upward drivers only push confidence up. Without decay, a synapse that got confident early on a spurious correlation freezes forever. Decay keeps the system able to adapt. |
| **Plain SGD, not Adam** | Adam already adapts a per-parameter learning rate; it would fight the confidence-gated rate. The confidence factor *is* our per-synapse rate adaptation. |
| **New synapses born at weight 0, confidence 0** | Zero weight ⇒ a newborn synapse is a no-op the instant it appears, so growth never disrupts learned behavior. Zero confidence ⇒ it is born at maximum plasticity. Result: stable consolidated core, plastic growing frontier — for free. |
| **Moderate sparsity, NOT brain-level** | Literal brain connectivity (~1e-7) is far too sparse for gradients to flow. Use ~40–70% sparsity. Brain-level is aspirational framing, not a v1 target. |
| **Grace period before pruning** | A freshly grown synapse must survive a minimum age before it is eligible to be pruned, or it dies before it can prove useful. |

---

## 3. The data structures

Do **not** model this as weight matrices. Model it as **two lists of objects**, because the per-synapse state is where the novelty lives and it's what gets visualized.

### Neuron
```
Neuron:
    id            : int
    layer         : int             # 0 = input ... L = output
    pos           : (x, y)          # fixed 2D coordinate for drawing
    activation    : float           # a_j, from the most recent forward pass
    firing_rate   : float           # r_j, EMA of activation (homeostasis + underfire detection)
    bias          : float           # init 0; trained by backprop
```

### Synapse
```
Synapse:
    pre           : int             # source neuron id
    post          : int             # target neuron id (must be in a later layer)
    weight        : float           # w, trained by backprop
    confidence    : float           # c >= 0, init 0   <-- novel
    eligibility   : float           # e >= 0, init 0   <-- novel ("fired together recently")
    age           : int             # steps since birth, init 0 (for grace period)
```

Store synapses in a structure that lets you (a) iterate all live synapses, (b) for a given neuron, fetch its incoming synapses (forward pass + homeostasis) and outgoing synapses (backprop). A dict keyed by `(pre, post)` plus per-neuron in/out adjacency lists works well.

---

## 4. The math (exact formulas)

Notation: `a` = activation, `r` = firing-rate EMA, `w` = weight, `c` = confidence, `e` = eligibility. Subscripts: `pre`/`post` for a synapse's endpoints, `j` for a neuron.

### 4.1 Forward pass
Process neurons in layer order. For each non-input neuron `j`:
```
z_j = bias_j + Σ_{live synapses i→j}  w_ij · a_i
a_j = ReLU(z_j)
```
- Input neurons: `a = (x, y)` components directly.
- Output layer: 2 neurons, apply **softmax** over their `z` values → class probabilities. (Or 1 neuron + sigmoid; softmax-over-2 is cleaner to extend.)

### 4.2 Firing-rate EMA (bookkeeping, every step)
```
r_j ← (1 − β) · r_j + β · a_j           # β ≈ 0.05
```

### 4.3 Eligibility trace update (every step, all live synapses)
```
coact = a_pre · a_post                  # rate-space "fired together"
e_ij  ← λ_e · e_ij + (1 − λ_e) · coact   # λ_e ≈ 0.9 (decay)
```

### 4.4 Backward pass (every step)
Standard backprop of the loss (cross-entropy on the softmax output). Compute `∂L/∂w_ij` **only for synapses that exist** and `∂L/∂bias_j` for all neurons. Gradients flow only along live edges. Use ReLU's derivative as usual.

### 4.5 Confidence update (every step) — the three-factor rule
Let `g = exp(−L)` be a global outcome-quality signal in `(0, 1]` (≈1 when loss is low). Then per live synapse:
```
c_ij ← (1 − γ_dec) · c_ij  +  e_ij · (γ_q · g + γ_h)
c_ij ← max(c_ij, 0)
```
Reading it in plain terms:
- `(1 − γ_dec)·c` — slow decay, so confidence can fall.
- `e_ij · (...)` — **eligibility gates credit**: only co-active synapses gain confidence (biologically sensible, and it unifies both drivers).
- `γ_q · g` — the *low-error* driver: gain more confidence when the global prediction is good.
- `γ_h` — the baseline *Hebbian* driver: gain a little confidence just for co-firing.

*Decoupled alternative (if you prefer the two drivers fully separate):* `c ← (1−γ_dec)·c + γ_q·g·𝟙[active] + γ_h·e`. The unified form above is the default.

*Optional v1.5 — signed/"punishing" confidence:* replace `g` with `(g − 0.5)` so bad outcomes on active synapses **reduce** confidence (a rate-space stand-in for LTD, since true post-before-pre timing isn't available). Try only after v1 works.

### 4.6 Confidence-gated weight update (every step)
```
η_eff_ij = η_base / (1 + c_ij)
w_ij ← w_ij − η_eff_ij · ∂L/∂w_ij
bias_j ← bias_j − η_base · ∂L/∂bias_j     # biases use base rate, ungated
```
Confident synapses barely move; unsure ones move fast. This is the signature mechanic.

### 4.7 Structural plasticity (every `T_struct` steps)

**Prune.** Utility of a synapse:
```
u_ij = |w_ij| · r_pre
prune  if  (u_ij < θ_prune)  AND  (age_ij > T_grace)
```
Cap the number pruned per round (`max_prune`) so changes stay gentle and watchable.

**Grow.** A neuron is *underfiring* if `r_j < f_under · r_target`. For each underfiring neuron `j` (capped at `max_grow`):
1. Consider candidate sources `i` such that: `layer(i) < layer(j)`, no synapse `i→j` already exists.
2. Pick the candidate with the **highest activation in the most recent forward pass** (the neuron that "just fired strongly before" `j`).
3. Create synapse `i→j` with `weight = 0, confidence = 0, eligibility = 0, age = 0`.

Increment `age` on every live synapse each step.

### 4.8 Homeostasis (optional, every `T_homeo` steps)
Gently push each neuron toward the target firing rate by scaling its incoming weights:
```
for neuron j:
    scale = (r_target / (r_j + ε)) ** ρ        # ρ ≈ 0.1 (gentle), ε ≈ 1e-6
    for each incoming synapse i→j:  w_ij ← w_ij · scale
```
This stabilizes the system, *defines* "underfiring" principledly, and keeps the initialization variance honest as `in-degree` changes. Add it only if training is unstable without it.

---

## 5. Initialization

### 5.1 The connection graph (which edges exist)
- Layered: input → hidden layers → output.
- For each neuron, connect it to a random subset of the previous layer at the chosen **density** (start ~0.5, i.e. ~50% of possible incoming edges).
- **Guarantee** `min fan-in ≥ 2` and `min fan-out ≥ 1` per neuron — pure random sampling strands neurons at higher sparsity.
- (Optional flavor) small-world wiring — mostly local plus a few long-range edges — if you want shorter path lengths at low density.

### 5.2 Weights — scale by *actual* sparse fan-in
For neuron `j` with live in-degree `k_j`:
```
w_ij ~ Normal(0, sqrt(2 / k_j))      # He init, but using the REAL number of inputs
```
This is the most common failure point: He/Xavier-initializing a dense matrix and masking it sparse assumes `n` inputs but only `k ≈ density·n` remain, so per-layer variance collapses and deep signal dies. Scale to the real `k_j`, per neuron, since the graph is irregular.

**Robust shortcut (recommended for an irregular graph):** use **LSUV** instead of the closed form — forward one batch, measure each layer's activation variance, rescale weights to unit variance, repeat once or twice. It's data-driven and doesn't care how weird the topology is.

### 5.3 Everything else
- `bias = 0` for all neurons.
- `confidence = 0`, `eligibility = 0`, `age = 0` for all synapses ⇒ early training is plain SGD until evidence accumulates.
- `firing_rate` initialized to 0 (or to `r_target` to avoid spurious early "underfiring").

---

## 6. The data / task

- **Input:** 2D point `(x, y)`. **Output:** 2 classes.
- **Start easy:** two Gaussian blobs (linearly separable-ish) to debug the pipeline.
- **Real test:** **two interleaving spirals** — hard enough that the network must genuinely use its capacity (so growth/pruning matter), easy enough that you know what success looks like and can watch the decision boundary form.
- ~500–1000 generated points. **Single-sample SGD** is fine and the most watchable; small mini-batches also fine.

---

## 7. The training loop

```
init data, network                       # §5, §6
for step in range(N_steps):
    x, y_true = sample()

    a            = forward(x)             # §4.1
    update_firing_rates(a)               # §4.2
    update_eligibility(a)                # §4.3
    L, grads     = backward(a, y_true)   # §4.4
    update_confidence(L)                 # §4.5
    apply_gated_update(grads)            # §4.6
    increment_ages()

    if step % T_struct == 0:             # §4.7
        prune()
        grow()
    if step % T_homeo == 0:              # §4.8 (optional)
        homeostasis()
    if step % T_draw == 0:
        render()                          # §9
```

Write each numbered step as its own function so each can be watched/tested in isolation.

---

## 8. Hyperparameters (starting values — tune from here)

| Name | Symbol | Start | Role |
|---|---|---|---|
| Base learning rate | `η_base` | 0.05 (single-sample) | weight update size |
| Eligibility decay | `λ_e` | 0.9 | memory length of "fired together" |
| Confidence decay | `γ_dec` | 0.001 | lets confidence fall |
| Confidence quality gain | `γ_q` | 0.05 | low-error → confidence |
| Confidence Hebbian gain | `γ_h` | 0.01 | co-firing → confidence |
| Firing-rate EMA | `β` | 0.05 | smoothing of activity |
| Target firing rate | `r_target` | 0.15 | homeostasis / underfire set point |
| Underfire fraction | `f_under` | 0.5 | grow if `r_j < f_under·r_target` |
| Prune threshold | `θ_prune` | 0.01 | utility floor (tune empirically) |
| Grace period | `T_grace` | 200 steps | min age before prunable |
| Structural interval | `T_struct` | 100 steps | how often prune/grow runs |
| Homeostasis interval | `T_homeo` | 50 steps | how often scaling runs |
| Max prune / round | `max_prune` | 2 | keep changes gentle |
| Max grow / round | `max_grow` | 2 | keep changes gentle |
| Init density | — | 0.5 | fraction of possible edges present |
| Draw interval | `T_draw` | 25 steps | render cadence |

**Network size:** input = 2 neurons; hidden = `[8, 8, 6]`; output = 2 (softmax). ~26 neurons total. Small enough that every neuron is a dot and every synapse is a visible line.

---

## 9. Visualization spec (the payoff)

Render the two object-lists every `T_draw` steps. Recommended: **matplotlib** (`FuncAnimation`) for simplicity and easy decision-boundary plotting; pygame is an alternative for smoother interaction.

**Main panel — the network graph:**
- **Neurons = dots** at their fixed `pos`, arranged in layer columns. **Size/brightness ∝ activation** `a_j`. You should see activity ripple left→right each pass.
- **Synapses = lines** between dots. **Thickness ∝ |weight|**. **Color ∝ confidence** via a colormap (e.g. blue = low/unsure/fast-learning → red = high/confident/frozen).
- A **line appearing = growth**; a **line vanishing = a prune**. The network visibly rewires.

**Side panels:**
- Classification **accuracy** vs step.
- Total **synapse count** vs step (watch it grow, then settle).
- **2D decision boundary**: evaluate the net on a grid over the input plane, color by predicted class; overlay the data points. Watch it bend to fit the spirals.

**The story that validates the idea:** lines start thin and blue everywhere and the boundary is garbage → key pathways thicken and redden as they consolidate → dead branches get pruned → underused neurons sprout fresh blue connections → the boundary snaps to the spiral shape. If you can see that sequence, the mechanics work.

---

## 10. Build order (do NOT build it all at once)

Add one mechanism at a time; confirm each is visibly working before adding the next. If a later step breaks something, you know exactly which mechanism caused it.

1. **Sparse net + backprop only.** Fixed sparse graph learns the *blobs*. Get the drawing working. ✅ when accuracy climbs and signal doesn't die in deep layers (validates §5.2 init).
2. **Add eligibility traces.** Don't change learning yet — just visualize traces glowing on co-active synapses. ✅ when traces look sane.
3. **Add confidence + gated learning rate.** ✅ when lines change color over time and confident (red) synapses visibly stop moving.
4. **Add pruning.** ✅ when low-utility lines disappear and accuracy doesn't collapse.
5. **Add growth.** ✅ when underfiring neurons sprout new (blue, thin) connections. *This is the climax.*
6. **Add homeostasis** only if unstable. ✅ when firing rates settle near `r_target`.

Then switch the dataset from blobs to **spirals** as the real test.

---

## 11. Validation criteria (what "it works" means)

The project succeeds — independent of accuracy — if you can demonstrate, on screen:
- [ ] Activations propagate and the decision boundary fits the spirals.
- [ ] Eligibility traces are high on co-active synapses, low elsewhere.
- [ ] Confidence rises on consistently useful, co-active synapses, and their effective learning rate visibly drops (`η_base/(1+c)`).
- [ ] Confidence can fall (decay) when a previously-confident synapse stops being reinforced.
- [ ] Low-utility synapses get pruned without tanking accuracy.
- [ ] Underfiring neurons grow new connections to recently-active upstream neurons; new synapses start at zero weight (no disruption) and are maximally plastic.
- [ ] Total synapse count grows then **stabilizes** (system finds an equilibrium, not runaway growth or total collapse).

---

## 12. Suggested code layout

```
sprout/
  data.py          # generate_blobs(), generate_spirals()
  network.py       # Neuron, Synapse, Network; build_graph(), init_weights() (§5), forward() (§4.1), backward() (§4.4)
  learning.py      # update_firing_rates(), update_eligibility(), update_confidence(), apply_gated_update()  (§4.2–4.6)
  plasticity.py    # prune(), grow(), homeostasis(), increment_ages()  (§4.7–4.8)
  viz.py           # render(): graph panel + accuracy + synapse-count + decision boundary  (§9)
  train.py         # hyperparameters (§8) + main loop (§7)
```

Pure NumPy is enough and keeps the per-synapse object model transparent. PyTorch is fine if you prefer autograd for §4.4, but the irregular sparse graph means you'll likely hand-roll the forward/backward over adjacency lists anyway — don't reach for dense layers.

---

## 13. Known pitfalls

- **Dead deep signal** → you forgot per-neuron fan-in scaling (§5.2). Use the real `k_j`, or LSUV.
- **Everything freezes** → confidence decay `γ_dec` too small or quality gain too large; confidence saturates. 
- **Runaway growth or mass extinction** → tune `θ_prune`, `T_grace`, and the per-round caps; consider enabling homeostasis.
- **Growth amplifies already-dominant pathways** → expected tendency; homeostasis (§4.8) counteracts it.
- **Visualization too busy to read** → increase `T_draw`, lower the per-round structural caps, shrink the network. Legibility is the whole point.
- **Don't scale up.** A 500-neuron net teaches you nothing you can see. Resist it for v1.

---

## 14. Explicitly out of scope for v1 (parked for later)

- Spiking neurons + surrogate-gradient learning (true STDP timing).
- Recurrent connections (would break the DAG/backprop simplicity).
- Gradient-informed growth (RigL-style: grow where the backprop gradient is largest, instead of/alongside activity-based growth).
- Confidence-gated exploration noise (Thompson-sampling-like behavior at the synapse level).
- A "sleep"/consolidation replay phase that reinforces high-confidence pathways offline. **— now implemented** as *phasic-startle-k4 structural plasticity*: the net rewires (prunes the weak + grows the wanted) only when the loss has settled, and hires on sustained loss spikes with a k=4 demand-bounded grow scan.
- Real datasets, real sparsity levels, wall-clock performance.
## Phasic-startle-k4 structural plasticity (baseline default)

Structural change is **phasic** and on by default (`Config.phasic_structure =
True`), in **three phases driven by one smoothed-loss state machine**: the net
**learns while awake** (confidence + gated-SGD + gradient metering every step,
*no* rewiring), **rewires while it sleeps**, and **hires when startled**
(`Config.startle = True`, default since 2026-06-12 — see below). The baseline is
**phasic-startle-k4**: startle growth uses `Config.grow_demand_k = 4`, so each
grow pass scores ghosts only into the top-4 highest-demand post neurons instead
of scanning every active pre/post pair. The old full grow scan remains available
with `grow_demand_k=None` and is pinned in the `phasic-startle` eval comparator.
"Sleep" is a **settledness** plateau, detected by an EMA of the training loss
(`sprout/sleep.py`, `SettlednessDetector` — early-stopping-style "patience" on the
smoothed loss, past a warmup). Loss is the signal because it is the only clean
one: mean confidence is non-monotonic and the gradient meter `M̄` is too spiky
(both measured).

On a settled step the trainer runs **one rewire pass** (`_rewire_phasic`): prune
every wire below `sleep_prune_floor` AND grow every ghost the loss wants above the
relative bar — both uncapped, the floor and the bar are the only limits — then
`reset()` the detector so a *fresh* plateau is required before the next pass.
Because rewires are far apart and gated on re-settling, grow↔prune oscillation is
**structurally impossible**: a just-cut wire cannot be re-requested mid-run, so
the older anti-oscillation patches (the inflated grow bar, the ghost-meter
refractory) stop being load-bearing here. Cold start is *pure phasic* — no
structural change until the first plateau, so the net learns on its initial graph
until then. This **subsumes** the earlier continuous-churn + sleep-overlay design;
the continuous path is retained behind `phasic_structure=False` as the pinned A/B
baseline. The lever is *timing*: one-shot rewiring of a *settled* net is ~lossless
where the same sparsity reached by continuous online churn craters accuracy.

**Prune floor = 1.0, no cap** (`sleep_prune_floor=1.0`, `sleep_max_prune=None`).
An uncapped floor sweep from 0→2 found accuracy holds with no clear regression all
the way to floor ~1.8 and then a sharp cliff at 2.0. The reason: the **floor is a
quality filter** — at 1.0 it sits *below* the median wire utility (~1.7), so only
the genuinely-weak tail is ever eligible — while the per-burst **cap is merely a
rate limit**. At matched sparsity, low-floor/no-cap beats high-floor/capped (it
keeps the *right* wires).

**Measured** (5 seeds, w16, 15k + 3k shift, `docs/eval-runs/phasic-vs-continuous`):
vs the continuous `currency` baseline, phasic **preserves accuracy** (final /
recovered test acc ≈), ends **~47% sparser** (synapse_count 234 → 123; mean fan-in
4.7 → 2.5, so ~2–3× cheaper forward + step — both are `O(live edges)`, and SPROUT
stores explicit synapses so fewer wires = fewer operations), with near-zero
re-grow (`max_regrow` 3.4 → 0.4) and higher-quality survivors (`p10_utility` ▲). It
also beats the old continuous+sleep default on accuracy, dead units and
oscillation at comparable sparsity.

**Startle: the hiring alarm (default since 2026-06-12).** Plateau-gated rewiring
is **demand-blind**: the grow signal (ghost virtual gradients) is loudest during
a transition's loss spike, which the settledness gate by construction waits out —
measured consequences were conservative growth and a second-task acquisition gap
vs the continuous baseline (`b_learned` 0.973 vs 0.983, `docs/eval-runs/
recycle-continual`). The startle is the mirror threshold on the same detector: a
**grow-only pass fired immediately** (any step, off the `t_struct` grid) once the
loss spike alarm trips, then `reset()` re-baselines the detector (refractory +
fixes a false-settled-mid-descent artifact). The alarm requires three conditions
together, each defeating a measured false-positive source: a **relative rise**
(fast EMA > slow `beta/10` EMA × 1.5 — *not* vs `best`, a min-ratchet that one
lucky noise trough poisons: 29 false alarms/run), an **absolute trouble floor**
(auto `ln(K)/2`, half chance-level CE — mid-training convergence waves and
post-burst prune bumps are up to ~20× *relative* but only ~0.1–0.17 absolute),
and **persistence** (50 consecutive steps). Result: **0 false alarms on
stationary runs** (the default is inert until the world changes), ~2 per
transition, firing ~60 steps in while `b_steps_to_80` ≈ 200. Measured
(`docs/eval-runs/startle-vs-phasic` + `startle-continual`): idle capacity
recruited (`idle_unit_frac` 0.30 → 0.22 ▲), churn down (`turnover` ▲), and the
continual worst seed raised 0.960 → 0.977 at end sparsity identical to sleep-only
phasic — but the *mean* `b_learned` gain is ≈ at 5 seeds (the residual gap vs
continuous is **refinement-tail growth**, not onset timing), and emergency hires
read as freeloaders until a sleep cycle cleans them. Sleep-only phasic is
retained behind `startle=False` (pinned in the eval `phasic`/`phasic-recycle`
variants and `validate.py`).

**Demand-bounded growth: the k4 default.** The promoted baseline adds one bound
to startle: `grow_demand_k=4`, meaning candidate scoring only considers the
top-4 highest-|delta| post neurons for each active pre neuron. This preserves the
same gradient-currency signal while replacing a broad active-pre/active-post
scan with a small demand frontier. Measured against full-scan `phasic-startle`
(5 seeds, w16): on the shift guardrail, scored ghost pairs fell **80.97 →
13.02**, grow events **60.6 → 15.6**, end synapses **179 → 141**, and turnover
**0.878 → 0.671 ▲**, with final and recovered test accuracy ≈ (**0.967 →
0.964**). On continual A→B→A+B, scored ghost pairs fell **84.71 → 15.86** and
end synapses **106 → 85**, with final accuracy ≈ and `b_learned` ≈ (**0.979 →
0.973**). The tested aroused window (`arousal_steps=1000`) is **not** promoted:
it added refinement activity but no clear second-task or consolidation win.
Recycling remains rejected. The settled simple architecture is therefore:
phasic sleep, startle hiring, and a k=4 demand-bounded grow scan.

**Honest caveats.** (1) Phasic cuts *edges*, not *neurons* and not the firing
fraction — the ~47% of neurons that fire per input is unchanged (synapse sparsity
⊥ activation sparsity); the compute win is lower fan-in. (2) It roughly **doubles
the dead-unit fraction** (~0.09 → ~0.18) and slightly worsens post-shift accuracy
stability — capacity that is permanently silent. The dead-unit cost **was
measured on the continual regime** (`docs/eval-runs/recycle-continual`): the
continuous baseline keeps ~2× more capacity in service and wins second-task
acquisition (`b_learned` ▲), so phasic's sparsity is not free under task
transitions; startle (above) recovers about half that gap. A sleep-time
*recycling* re-entry path for dead units was tried and **rejected** (zero
rehires — a blank's bid cannot out-compete live features; see the 2026-06-11
recycling spec). `validate.py` and the eval `currency`/`sleep` baselines pin
`phasic_structure=False` as fixed references.

## Research roadmap: compute efficiency

The grow scan's `O(N²)` ghost enumeration was first replaced by an
activity-sparse scan (see `sprout/currency.py` `active_ghost_sets` /
`iter_ghost_candidates` and the `docs/eval-runs` grow-scan scaling run), then the
baseline narrowed that scan to the top-4 demanded post neurons
(`grow_demand_k=4`). This makes growth proportional to `k * active_pre` during
the rare sleep/startle passes, while the normal per-step path is still
`O(live edges)`. Two further sparsity wins are deferred — both improve wall-clock
constant factors, but they are not needed for the current architectural
promotion:

**2026-06-12 compute-efficiency audit.** The eval harness now reports wall time,
total train edge-steps, edge-steps-to-threshold, and active-edge fractions. At
w16, `phasic-startle-k4` keeps fully-connected accuracy while using far fewer
edges: **140.8 vs 576** final synapses, **3.19M vs 8.64M** train edge-steps, and
lower measured training-loop wall time (`compute-efficiency-finalists`). Fully
connected still wins raw SGD steps to 90/95%, so sparse training is only cheaper
when measured as edge work or wall time, not as step count.

The obvious cheaper probe, `eff-density30`, is **not promoted**. It is faster and
sparser on stationary spirals (fixed-horizon train edge-steps **2.67M vs 3.19M**),
but the shift guardrail exposes the robustness cost: recovered accuracy **0.964
→ 0.923**, dead-unit fraction **0.150 → 0.208**, and worse steps-to-95
(`compute-efficiency-density30-shift`). Smaller width (`eff-w12`) saves fixed
edge work but learns too slowly; lazy gradient meters are exact but slower because
confidence/update still scan live wires; top-k/WTA reduces active fractions only
modestly unless made aggressive, and aggressive WTA hurts accuracy. Conclusion:
keep `phasic-startle-k4` at density 0.4 as the robust compute baseline.

- **#2 Lazy gradient meters.** `update_gradient_meters` EMA-updates *all* live
  wires every step, but a wire whose pre was silent has gradient 0, so its update
  is pure decay `M ← β·M`. This is now implemented as an opt-in exact variant
  (`lazy_meters=True`, `phasic-startle-k4-lazy`): each wire tracks its last meter
  step and applies `β^Δ` on access. Result: **not promoted**; the extra access
  logic costs more than it saves while confidence and weight-update loops still
  visit all live wires.
- **#3 Event-driven forward/backward.** `forward`/`backward` visit every hidden
  neuron each step. An active-frontier formulation propagates only out of neurons
  that fired (ReLU output ≠ 0), so silent neurons and their wires cost nothing —
  turning the per-step path from `∝ edges` into `∝ active edges`. The largest
  inner-loop change; still deferred. The new WTA probes (`eff-wta4/6/8`) show the
  trade-off: mild top-k barely moves wall time, while aggressive top-k damages
  accuracy. Event-driven execution should wait until activation sparsity is both
  enforced and accuracy-safe.

### #3 in detail: the per-sample active subgraph

**Mechanism.** Both passes currently iterate every neuron and its full adjacency
(`sprout/network.py` `forward` / `backward`). The event-driven rewrite tracks a
*frontier* and touches only the edges that carry real signal:

- *Forward (a "fired" frontier).* Initialise each non-input neuron's
  pre-activation to its bias (`z_j ← bias_j`); input neurons are the seed
  frontier. Walking layers in order, each fired neuron *scatters* `w_ij·a_i` into
  its posts' `z`; after a layer finishes, one pass applies ReLU and the neurons
  with `z_j > 0` become the next frontier. Work = `O(N)` for the bias/ReLU pass
  plus `Σ out-degree` over *fired* neurons — i.e. `∝ fwd-active edges`, where an
  edge is fwd-active iff its pre fired (`a_pre ≠ 0`). The `O(N)` floor is
  irreducible: every neuron must still be ReLU'd and bias-initialised, and a
  neuron with `bias_j > 0` fires even with an all-silent fan-in, so it joins the
  frontier on bias alone.
- *Backward (a "nonzero-delta" frontier).* `δ_j = relu'(z_j)·Σ w_jk·δ_k` is zero
  whenever `j` was silent (`relu'=0`), so silent neurons neither receive nor
  propagate a delta. Seed the frontier with the output neurons (always nonzero),
  and scatter `w_pre,j·δ_j` only *back through fired pres* (a silent pre's δ is
  killed by its own `relu'` regardless). Work `∝ bwd-active edges`.

**The unifying insight — one active subgraph for the whole step.** The weight
gradient is `grad_w[(pre,post)] = δ_post · a_pre`, which is nonzero **iff
`a_pre ≠ 0` AND `δ_post ≠ 0`** — the *doubly-active* edges. Every per-step
consumer of the gradient is therefore already structurally sparse on the same
subgraph: `apply_gated_update` changes a weight only on doubly-active edges (a
zero-gradient edge is a pure no-op, not even decay), and `update_gradient_meters`
does real work only there (everywhere else it is the decay `M ← β·M` that #2
defers). So #2 and #3 are two views of one optimisation: **confine forward,
backward, the meter update, and the weight update to the per-sample active
subgraph.** Build the frontier once in forward, reuse it everywhere downstream.

**Measured magnitude (this is the important part).** The ceiling is set purely by
realised ReLU sparsity, and in the v1 nets that sparsity is *mild* — far from
what `r_target = 0.15` suggests, because nothing in currency mode enforces it
(homeostasis is off; growth is gradient-driven, not firing-rate-driven). On the
**default w16 net (`(2,16,16,16,2)`, currency, spirals, 15k steps)**, averaged
over the held-out test set:

| quantity | value | ceiling for |
|---|---|---|
| hidden neurons firing (`a > 0`) | **~44%** | — |
| fwd-active edges (pre fired) | ~60% | forward **~1.7×** |
| bwd-active edges (post δ ≠ 0) | ~57% | backward **~1.7×** |
| doubly-active edges (grad ≠ 0) | ~36% | meters + update **~2.8×** |

A width/depth sweep (w6→w48, plus an 8-hidden-layer net) shows the firing
fraction stays roughly flat at **~40–47%** as width grows (so this is a
**constant-factor** win, not asymptotic — the active *fraction* doesn't shrink
with `N`), and improves only modestly with depth (the deep net fires ~39%, for a
~2.3× forward / ~4.9× gradient ceiling).

**The catch, and the prerequisite.** Those are *zero-overhead ceilings*. The
current inner loop is Python scalar ops with tiny constants; an event-driven
rewrite adds frontier bookkeeping (membership sets, the scatter accumulator, the
unavoidable `O(N)` ReLU/bias pass). At the v1 default scale (`N=52`, `E≈250`) that
overhead would likely **erase or invert** a 1.7× edge-count saving — #3 is a wash
here. It pays off only when one of these holds: **(a)** nets get much larger, so
`E` dwarfs the `O(N)` floor and the constant amortises; **(b)** activation
sparsity is actually *enforced* — turning on homeostasis toward `r_target`, or a
k-winners-take-all activation, would push firing toward ~15% and lift the forward
ceiling to ~6× (and under fixed-`k` WTA the active fraction shrinks with width,
making #3 genuinely *asymptotic*); or **(c)** the inner loop leaves Python for a
vectorised/compiled backend where branchy frontier logic is cheap. The honest
ordering: **#3's payoff is gated on activation sparsity being real, and right now
it isn't** — so the highest-leverage precursor is to enforce (or at least measure
under homeostasis) ReLU sparsity before investing in the event-driven rewrite.
