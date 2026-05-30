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
    lambda_e: float = 0.9         # eligibility decay (memory of "fired together")
    # Confidence gains - retuned from the spec so consolidation LAGS learning
    # (see update_confidence's deviation note). gamma_dec is larger so frozen
    # synapses still creep and decay can release them; eligibility enters as a
    # bounded gate with half-saturation e_half.
    gamma_dec: float = 0.01       # confidence decay (lets confidence fall)
    gamma_q: float = 0.05         # confidence quality gain (low error -> confidence)
    gamma_h: float = 0.005        # confidence Hebbian gain (co-firing -> confidence)
    e_half: float = 0.1           # eligibility gate half-saturation
    beta: float = 0.05            # firing-rate EMA smoothing
    r_target: float = 0.15        # target firing rate
    f_under: float = 0.5          # grow if r_j < f_under * r_target
    theta_prune: float = 0.01     # utility floor for pruning
    t_grace: int = 200            # min age before a synapse is prunable
    t_struct: int = 100           # how often prune/grow runs
    prune_warmup: int = 0         # don't prune before this step (let weights develop)
    t_homeo: int = 50             # how often homeostasis runs
    max_prune: int = 2            # cap pruned per round (gentle, watchable)
    max_grow: int = 2             # cap grown per round
    grow_budget: int = 6          # max growth attempts per neuron (retires dead units)
    rho: float = 0.1              # homeostasis gentleness
    eps: float = 1e-6

    # --- feature flags (build order, §10) ---
    enable_eligibility: bool = False
    enable_confidence: bool = False
    enable_prune: bool = False
    enable_grow: bool = False
    enable_homeostasis: bool = False

    # --- gradient-as-currency mode (enhancement; replaces eligibility path) ---
    # When True, confidence/prune/grow read the per-synapse gradient meters
    # instead of eligibility / |w|*r / activity. See sprout/currency.py.
    grad_currency: bool = False
    beta_g: float = 0.99          # gradient-meter EMA memory (~100 samples)
    gamma_up: float = 0.05        # confidence earn rate (calm + consistent)
    gamma_dn: float = 0.10        # confidence loss rate (contested hot-spot)
    c_max: float = 5.0            # confidence ceiling (legible band)
    m_floor_frac: float = 0.05    # "no real feedback" cutoff for kappa, * Mbar
    lam_prune: float = 1.0        # gradient weight in prune utility
    prune_u_floor: float = 0.5    # prune wires with normalised utility below this
    grow_bar_frac: float = 1.5    # grow ghost wire if virt-grad > this * live ref
    virt_batch: int = 32          # batch size for scoring ghost wires

    # --- initial firing rate seeding ---
    init_firing_rate_at_target: bool = True  # avoids spurious early "underfiring"


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

        if cfg.grad_currency:
            loss, grad_w, grad_b = net.backward(y_true)
            n_pruned, n_grown = self._step_currency(grad_w, grad_b)
        else:
            if cfg.enable_eligibility:
                self._update_eligibility()               # §4.3
            loss, grad_w, grad_b = net.backward(y_true)  # §4.4
            if cfg.enable_confidence:
                self._update_confidence(loss)            # §4.5
            apply_gated_update(net, grad_w, grad_b, cfg.eta_base)  # §4.6
            self._increment_ages()

            n_pruned = n_grown = 0
            if (cfg.enable_prune or cfg.enable_grow) and self.step_idx % cfg.t_struct == 0:
                if cfg.enable_prune and self.step_idx >= cfg.prune_warmup:
                    n_pruned = self._prune()
                if cfg.enable_grow:
                    n_grown = self._grow()
            if cfg.enable_homeostasis and self.step_idx % cfg.t_homeo == 0:
                self._homeostasis()

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

    def _update_eligibility(self):
        from sprout.learning import update_eligibility
        update_eligibility(self.net, self.cfg.lambda_e)

    def _update_confidence(self, loss):
        from sprout.learning import update_confidence
        update_confidence(self.net, loss, self.cfg.gamma_dec, self.cfg.gamma_q,
                           self.cfg.gamma_h, self.cfg.e_half)

    def _increment_ages(self):
        for syn in self.net.synapses.values():
            syn.age += 1

    def _prune(self):
        from sprout.plasticity import prune
        pruned = prune(self.net, self.cfg.theta_prune, self.cfg.t_grace, self.cfg.max_prune)
        for edge in pruned:
            self.events.append({"step": self.step_idx, "type": "prune", "edge": edge})
        return len(pruned)

    def _grow(self):
        from sprout.plasticity import grow
        grown = grow(self.net, self.cfg.r_target, self.cfg.f_under, self.cfg.max_grow,
                     self.cfg.grow_budget)
        for edge in grown:
            self.events.append({"step": self.step_idx, "type": "grow", "edge": edge})
        return len(grown)

    def _homeostasis(self):
        from sprout.plasticity import homeostasis
        homeostasis(self.net, self.cfg.r_target, self.cfg.rho, self.cfg.eps)

    # -- gradient-as-currency path (enhancement) ----------------------------
    def _step_currency(self, grad_w, grad_b):
        """One step in currency mode: meter the gradient, then read it three
        ways (confidence, prune, grow). Returns (n_pruned, n_grown)."""
        from sprout.currency import update_gradient_meters, update_confidence_currency
        cfg, net = self.cfg, self.net

        update_gradient_meters(net, grad_w, cfg.beta_g)             # §1 currency
        if cfg.enable_confidence:                                   # Readout A
            update_confidence_currency(net, cfg.gamma_dec, cfg.gamma_up,
                                       cfg.gamma_dn, cfg.c_max, cfg.m_floor_frac)
        apply_gated_update(net, grad_w, grad_b, cfg.eta_base)       # gated SGD
        self._increment_ages()

        n_pruned = n_grown = 0
        if (cfg.enable_prune or cfg.enable_grow) and self.step_idx % cfg.t_struct == 0:
            n_pruned, n_grown = self._rewire_currency()
        return n_pruned, n_grown

    def _rewire_currency(self):
        from sprout.currency import prune_currency, grow_currency, batch_edge_scores
        cfg, net = self.cfg, self.net
        n_pruned = n_grown = 0
        if cfg.enable_prune:                                        # Readout B
            pruned = prune_currency(net, cfg.t_grace, cfg.max_prune,
                                    cfg.prune_u_floor, cfg.lam_prune)
            for edge in pruned:
                self.events.append({"step": self.step_idx, "type": "prune", "edge": edge})
            n_pruned = len(pruned)
        if cfg.enable_grow:                                         # Readout C
            b = min(cfg.virt_batch, len(self.X))
            idx = self.rng.choice(len(self.X), size=b, replace=False)
            ghost, ref = batch_edge_scores(net, self.X[idx], self.y[idx])
            grown = grow_currency(net, ghost, ref, cfg.max_grow, cfg.grow_bar_frac)
            for edge in grown:
                self.events.append({"step": self.step_idx, "type": "grow", "edge": edge})
            n_grown = len(grown)
        return n_pruned, n_grown
