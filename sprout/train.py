"""Hyperparameters (§8) and the training loop (§7).

The :class:`Trainer` implements the per-step loop from §7, but every advanced
mechanism is behind a flag in :class:`Config` so we can follow the build order
in §10 - enable one mechanism at a time and confirm it works before adding the
next. With all flags off, :meth:`Trainer.step` is plain single-sample SGD.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sprout.learning import apply_gated_update


@dataclass
class Config:
    # --- core (§8) ---
    eta_base: float = 0.05        # base learning rate (single-sample)
    beta: float = 0.05            # firing-rate EMA smoothing (viz bookkeeping)
    r_target: float = 0.15        # target firing rate (firing-rate seeding only)
    t_grace: int = 200            # min age before a synapse is prunable
    t_struct: int = 100           # how often the rewire step runs
    max_prune: int = 2            # cap pruned per round (gentle, watchable)
    max_grow: int = 2             # cap grown per round
    eps: float = 1e-6

    # --- feature flags: the three currency readouts (toggle for ablations) ---
    enable_confidence: bool = False
    enable_prune: bool = False
    enable_grow: bool = False

    # --- gradient-as-currency: the only architecture now ---
    # DEPRECATED FLAG: the legacy non-currency path was removed, so the currency
    # readouts always run regardless of this flag. Retained transiently (default
    # True) so existing eval-variant definitions keep constructing; will be
    # dropped in a follow-up. See sprout/currency.py.
    grad_currency: bool = True
    beta_g: float = 0.99          # gradient-meter EMA memory (~100 samples)
    # tug-of-war confidence (opt-in alt rule, confidence_mode="tugofwar"):
    gamma_dec: float = 0.01       # confidence decay (lets confidence fall)
    gamma_up: float = 0.05        # confidence earn rate (calm + consistent)
    gamma_dn: float = 0.10        # confidence loss rate (contested hot-spot)
    c_max: float = 5.0            # confidence ceiling (legible band)
    m_floor_frac: float = 0.05    # "no real feedback" cutoff for kappa, * Mbar
    lam_prune: float = 1.0        # gradient weight in prune utility
    prune_u_floor: float = 0.5    # prune wires with normalised utility below this
    # Grow a ghost wire only if its virtual gradient exceeds this * the typical
    # live wire's gradient. Promoted 1.5 -> 3.0: a *selective* hiring bar (grow
    # only wires the loss wants much more than a typical live wire) is what fixes
    # the grow<->prune oscillation at its source — it shrinks how many wires
    # thrash and ends ~20% sparser, at no accuracy cost (docs/eval-runs/
    # b1-growbar-sweep). 1.5 was the prior eager default (variant currency-eager).
    grow_bar_frac: float = 3.0    # grow ghost wire if virt-grad > this * live ref
    virt_batch: int = 32          # batch size for scoring ghost wires
    # Phase-2 demand bound for the grow scan: when an int k, score ghosts only
    # into the top-k highest-|delta| post neurons (bounds work to k * active_pre).
    # None => exact-sparse scan over all active posts (the bit-identical default).
    grow_demand_k: int | None = None
    # A2 anti-oscillation: grow on a persistent EMA of the virtual gradient
    # (a "ghost meter") instead of one noisy batch. A just-pruned wire has no
    # meter entry, so it must re-earn growth over several cycles rather than being
    # re-requested on the next spike. Opt-in so the baseline is unchanged.
    ghost_meter: bool = False
    beta_ghost: float = 0.8       # ghost-meter EMA memory (higher = longer refractory)
    # confidence rule (currency mode): "twod" (importance x settledness,
    # calibrated to prune utility — the DEFAULT / baseline) or "tugofwar" (the
    # prior calm+consistent earn/lose rule, kept for comparison). See currency.py.
    confidence_mode: str = "twod"
    conf_gain: float = 2.0        # 2D confidence: above-average-weight -> confidence
    conf_alpha: float = 0.01      # 2D confidence: EMA rate toward the target
    # 2D confidence: how settledness reads relative demand d = M/Mbar. "hard" is
    # the original ReLU cliff (zero past average demand); the smooth modes keep a
    # contested load-bearer off zero. "sigmoid" leads (smooth version of the cliff).
    settled_mode: str = "sigmoid"  # "hard" | "sigmoid" | "exp" | "rational"
    conf_k: float = 3.0            # settled-cliff steepness

    # --- phasic structural plasticity (the C architecture; ON by default) ---
    # When True, the net changes structure ONLY at settledness plateaus: wake =
    # pure gated-SGD + meter the gradient (no prune/grow); sleep = one rewire pass
    # (prune the weak + grow the wanted), then re-settle. Subsumes the sleep
    # overlay and makes grow<->prune oscillation structurally impossible. When
    # False, the legacy CONTINUOUS path runs (gentle prune+grow every t_struct,
    # with the sleep overlay if enable_sleep) — retained as the pinned A/B
    # baseline + the validate.py guardrail. See the rewire step in Trainer.
    phasic_structure: bool = True

    # --- sleep consolidation (ON by default) ---
    # Prune AGGRESSIVELY only when the net has settled (a loss-EMA plateau),
    # instead of churning continuously. The lever is *timing*, not the criterion
    # (sleep reuses prune_currency). PROMOTED to the default at floor 1.0 / NO cap
    # (sleep_max_prune=None => each burst removes every below-floor wire): the
    # floor-0-to-2 sweep found this the deepest preserved-accuracy operating point
    # (~-46% synapses at preserved single-task accuracy). The floor stays *below*
    # the median wire utility (~1.7) so only the genuinely-weak tail is ever
    # eligible — the floor is the quality filter that keeps deep pruning safe.
    # validate.py and the eval `currency` baseline pin enable_sleep=False as stable
    # references. See sprout/sleep.py and docs/eval-runs/sleep-nocap-floor-0-to-2.
    enable_sleep: bool = True
    sleep_warmup: int = 2500       # no consolidation before this step
    sleep_loss_beta: float = 0.01  # loss-EMA smoothing (~100-step memory)
    sleep_loss_tol: float = 0.03   # rel. loss improvement that resets the plateau
    sleep_patience: int = 1500     # steps w/o a tol-improvement => settled
    sleep_prune_floor: float = 1.0  # prune utility floor during a burst (quality filter)
    sleep_max_prune: int | None = None  # per-burst cap; None = no cap (prune all eligible)

    # --- initial firing rate seeding ---
    init_firing_rate_at_target: bool = True  # avoids spurious early "underfiring"

    # --- eval-harness build hint (NOT read by Trainer) ---
    # Optional per-variant override of the *initial* graph density used by the
    # eval runner's build_graph. None => use the suite's --density. This lets a
    # single suite mix connectivity regimes — e.g. a fully-connected control
    # (init_density=1.0, all plasticity off) against the sparse, self-rewiring
    # currency baseline. The Trainer ignores it; only the runner consults it.
    init_density: float | None = None

    # Optional per-variant override of the *initial* layer sizes (neuron counts
    # per layer) used by the eval runner's build_graph. None => use the suite's
    # --layers. Sibling to init_density: it lets one suite hold the config fixed
    # while sweeping network size (e.g. a hidden-width sweep). Trainer ignores it.
    init_layers: tuple[int, ...] | None = None


def _softmax_predict_probs(net, X):
    return np.array([net.forward(x) for x in X])


def predict(net, X):
    return _softmax_predict_probs(net, X).argmax(axis=1)


def accuracy(net, X, y):
    return float((predict(net, X) == y).mean())


class Trainer:
    def __init__(self, cfg: Config, net, X, y, seed=0):
        self.cfg = cfg
        self.net = net
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y)
        self.rng = np.random.default_rng(seed)
        self.step_idx = 0
        self.history = {
            # logged every step
            "step": [],
            "synapse_count": [],
            # logged only on recorded steps (full-dataset accuracy is expensive)
            "rec_step": [],
            "accuracy": [],
            "loss": [],
            "n_pruned": [],
            "n_grown": [],
            "mean_confidence": [],
            "mean_eff_lr": [],     # mean of eta_base/(1+c): the gated rate
            "frac_frozen": [],     # fraction of synapses with c > 1
        }
        # optional per-synapse confidence trajectories (set via track_synapses)
        self.tracked = {}  # (pre,post) -> list of confidence over recorded steps
        # event log so the viz / tests can see structural changes as they happen
        self.events = []  # list of dicts: {"step", "type": "prune"|"grow", "edge"}
        # A2: persistent EMA of the virtual gradient per candidate (ghost) wire,
        # carried across rewire cycles so growth reacts to a sustained signal.
        self.ghost_meter = {}  # (pre, post) -> ema score; only when cfg.ghost_meter
        # sleep consolidation: detect when the net has settled, then prune hard.
        self.settled = False
        self.sleep_detector = None
        if cfg.enable_sleep:
            from sprout.sleep import SettlednessDetector
            self.sleep_detector = SettlednessDetector(
                cfg.sleep_loss_beta, cfg.sleep_loss_tol,
                cfg.sleep_patience, cfg.sleep_warmup)

        if cfg.init_firing_rate_at_target:
            for n in net.neurons:
                n.firing_rate = cfg.r_target

        # imported lazily so step-1 has no dependency on later modules
        self._learning = None
        self._plasticity = None

    def track(self, keys):
        """Register synapses whose confidence we log every recorded step."""
        for k in keys:
            self.tracked.setdefault(k, [])

    # -- per-step loop (§7) -------------------------------------------------
    def step(self, record=False):
        cfg = self.cfg
        net = self.net
        i = self.rng.integers(len(self.X))
        x, y_true = self.X[i], int(self.y[i])

        net.forward(x)                                   # §4.1
        self._update_firing_rates()                      # §4.2 (kept for viz)

        loss, grad_w, grad_b = net.backward(y_true)
        n_pruned, n_grown = self._step_currency(grad_w, grad_b, loss)

        # every-step bookkeeping
        self.history["step"].append(self.step_idx)
        self.history["synapse_count"].append(len(net.synapses))
        if record:
            # expensive full-dataset accuracy only on recorded steps
            self.history["rec_step"].append(self.step_idx)
            self.history["accuracy"].append(accuracy(net, self.X, self.y))
            self.history["loss"].append(loss)
            self.history["n_pruned"].append(n_pruned)
            self.history["n_grown"].append(n_grown)
            confs = [s.confidence for s in net.synapses.values()]
            self.history["mean_confidence"].append(float(np.mean(confs)) if confs else 0.0)
            eff = [cfg.eta_base / (1.0 + s.confidence) for s in net.synapses.values()]
            self.history["mean_eff_lr"].append(float(np.mean(eff)) if eff else cfg.eta_base)
            self.history["frac_frozen"].append(
                float(np.mean([c > 1.0 for c in confs])) if confs else 0.0)
            for key in self.tracked:
                syn = net.synapses.get(key)
                self.tracked[key].append(syn.confidence if syn else None)

        self.step_idx += 1
        return loss

    # -- §4.2 firing-rate EMA (always on; cheap bookkeeping) ----------------
    def _update_firing_rates(self):
        from sprout.learning import update_firing_rates
        update_firing_rates(self.net, self.cfg.beta)

    def _increment_ages(self):
        for syn in self.net.synapses.values():
            syn.age += 1

    # -- gradient-as-currency path (the only structural path) ---------------
    def _step_currency(self, grad_w, grad_b, loss):
        """One step in currency mode: meter the gradient, then read it three
        ways (confidence, prune, grow). Returns (n_pruned, n_grown)."""
        from sprout.currency import (update_gradient_meters,
                                      update_confidence_currency,
                                      update_confidence_2d)
        cfg, net = self.cfg, self.net

        update_gradient_meters(net, grad_w, cfg.beta_g)             # §1 currency
        if cfg.enable_confidence:                                   # Readout A
            if cfg.confidence_mode == "twod":
                update_confidence_2d(net, cfg.conf_gain, cfg.conf_alpha, cfg.c_max,
                                     cfg.settled_mode, cfg.conf_k)
            else:
                update_confidence_currency(net, cfg.gamma_dec, cfg.gamma_up,
                                           cfg.gamma_dn, cfg.c_max, cfg.m_floor_frac)
        apply_gated_update(net, grad_w, grad_b, cfg.eta_base)       # gated SGD
        self._increment_ages()
        if self.sleep_detector is not None:                        # settledness
            self.settled = self.sleep_detector.update(loss, self.step_idx)

        n_pruned = n_grown = 0
        if (cfg.enable_prune or cfg.enable_grow) and self.step_idx % cfg.t_struct == 0:
            n_pruned, n_grown = self._rewire_currency()
        return n_pruned, n_grown

    def _rewire_currency(self):
        from sprout.currency import (prune_currency, grow_currency,
                                      batch_edge_scores, update_ghost_meter)
        cfg, net = self.cfg, self.net
        n_pruned = n_grown = 0

        # Sleep consolidation: when the net has SETTLED, prune aggressively and
        # skip grow (don't explore while consolidating). Then require a fresh
        # loss plateau before the next burst, which makes churn impossible.
        consolidating = bool(cfg.enable_sleep and self.settled)
        if consolidating:
            self.events.append({"step": self.step_idx, "type": "sleep", "edge": None})

        if cfg.enable_prune:                                        # Readout B
            floor = cfg.sleep_prune_floor if consolidating else cfg.prune_u_floor
            cap = cfg.sleep_max_prune if consolidating else cfg.max_prune
            if cap is None:                     # sleep 'no cap' => all eligible
                cap = len(net.synapses)
            pruned = prune_currency(net, cfg.t_grace, cap, floor, cfg.lam_prune)
            for edge in pruned:
                self.events.append({"step": self.step_idx, "type": "prune", "edge": edge})
            n_pruned = len(pruned)
        if cfg.enable_grow and not consolidating:                   # Readout C
            b = min(cfg.virt_batch, len(self.X))
            idx = self.rng.choice(len(self.X), size=b, replace=False)
            ghost, ref = batch_edge_scores(net, self.X[idx], self.y[idx],
                                           grow_demand_k=cfg.grow_demand_k)
            if cfg.ghost_meter:        # A2: grow on the sustained EMA, not 1 batch
                update_ghost_meter(self.ghost_meter, ghost, cfg.beta_ghost)
                scores = self.ghost_meter
            else:
                scores = ghost
            grown = grow_currency(net, scores, ref, cfg.max_grow, cfg.grow_bar_frac)
            for edge in grown:
                self.events.append({"step": self.step_idx, "type": "grow", "edge": edge})
                self.ghost_meter.pop(edge, None)   # now live -> not a candidate
            n_grown = len(grown)

        if consolidating:                       # require a fresh plateau next time
            self.sleep_detector.reset()
            self.settled = False
        return n_pruned, n_grown
