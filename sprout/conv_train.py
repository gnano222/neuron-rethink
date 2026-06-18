"""Conv-SPROUT joint model + trainer (Phase 2).

Composes a weight-shared ``ConvEconomy`` front-end (sprout/conv.py) with the
EXISTING sparse phasic head (sprout.network.Network), and trains both economies
together: simultaneous wake (gradient-as-currency learning every step) and a
shared phasic plateau where both rewire (the head via the existing
currency.prune/grow, the conv via filter birth/death). Nothing in the validated
core changes — this module only orchestrates the existing primitives.
See docs/superpowers/specs/2026-06-16-conv-sprout-phase2-design.md.
"""
from __future__ import annotations

import math

import numpy as np

from sprout import currency, learning
from sprout.sleep import SettlednessDetector

_EPS = 1e-12


class ConvModel:
    """A ConvEconomy front-end + a head Network. The head's input layer reads the
    flattened pooled conv features (filter-major: filter, then pooled row/col)."""

    def __init__(self, conv, head, h, w):
        self.conv = conv
        self.head = head
        self.h, self.w = h, w
        self.input_ids = list(head.layers[0])
        assert len(self.input_ids) == conv.feat_dim(h, w), (
            "head input layer must match conv feature dim "
            f"({len(self.input_ids)} vs {conv.feat_dim(h, w)})")

    def forward(self, img):
        feat, cache = self.conv.forward(img)
        probs = self.head.forward(feat)
        return probs, feat, cache

    def input_delta(self, grad_b):
        """Loss gradient at each head INPUT neuron (= conv feature), which
        head.backward does not compute: dL/dfeat[p] = sum_post w(p,post)*delta_post.
        Bridges the head's backprop into the conv layer."""
        head = self.head
        d = np.zeros(len(self.input_ids))
        for p, iid in enumerate(self.input_ids):
            s = 0.0
            for post in head.outgoing[iid]:
                s += head.synapses[(iid, post)].weight * grad_b.get(post, 0.0)
            d[p] = s
        return d

    def predict(self, X_imgs):
        return np.array([self.forward(img)[0] for img in X_imgs]).argmax(axis=1)


class ConvTrainer:
    """Joint wake trainer for a ConvModel. Mirrors Trainer._step_currency for the
    head (forward -> meters -> 2D confidence -> gated update) and runs the same
    currency updates on the conv filters, bridged by the head input delta.

    Phasic structure (head + conv rewire at a settledness plateau) is added in
    Stage D; with structure off this is pure joint gated-SGD + metering.
    """

    def __init__(self, cfg, model, X_imgs, y, seed=0, conv_eta=None,
                 learn_conv=True, conv_structure=False, conv_k_max=None,
                 conv_grow_mode="split", conv_prune_floor=0.5, conv_k_min=2,
                 conv_grow_per_burst=2, conv_eta_schedule="none",
                 total_steps=None, freeze_frac=0.6,
                 conv_redundancy_prune=False, conv_redundancy_threshold=0.9,
                 conv_redundancy_mode="kernel", conv_redundancy_batch=32):
        self.cfg = cfg
        self.model = model
        self.X = np.asarray(X_imgs, dtype=float)
        self.y = np.asarray(y)
        self.rng = np.random.default_rng(seed)
        self.step_idx = 0
        self.conv_eta = cfg.eta_base if conv_eta is None else conv_eta
        # filter-learning-rate CONSOLIDATION: wind filter learning down over
        # training so filters settle near their peak and the head finishes on a
        # stationary representation (the stability fix; see findings doc).
        #   "none"   = constant (the unstable reference)
        #   "cosine" = base * 0.5*(1+cos(pi*step/total)) -> ~0 by the end
        #   "freeze" = base until freeze_frac of training, then 0 (learn-then-lock)
        self.conv_eta_schedule = conv_eta_schedule
        self.total_steps = total_steps
        self.freeze_frac = freeze_frac
        self.learn_conv = learn_conv
        self.conv_structure = conv_structure
        self.conv_k_max = model.conv.k_max if conv_k_max is None else conv_k_max
        self.conv_grow_mode = conv_grow_mode
        self.conv_prune_floor = conv_prune_floor
        self.conv_k_min = conv_k_min
        self.conv_grow_per_burst = conv_grow_per_burst
        self.conv_redundancy_prune = conv_redundancy_prune
        self.conv_redundancy_threshold = conv_redundancy_threshold
        self.conv_redundancy_mode = conv_redundancy_mode
        self.conv_redundancy_batch = conv_redundancy_batch
        self.events = []
        model.head.activation_top_k = cfg.activation_top_k
        self.detector = SettlednessDetector(
            cfg.sleep_loss_beta, cfg.sleep_loss_tol, cfg.sleep_patience,
            cfg.sleep_warmup)

    # -- wake ----------------------------------------------------------------
    def step(self, record=True):
        cfg, model = self.cfg, self.model
        head, conv = model.head, model.conv
        i = int(self.rng.integers(len(self.X)))
        img, yi = self.X[i], int(self.y[i])

        _, _, cache = model.forward(img)
        learning.update_firing_rates(head, cfg.beta)
        loss, gw, gb = head.backward(yi)
        currency.update_gradient_meters(head, gw, cfg.beta_g,
                                        step_idx=self.step_idx, lazy=cfg.lazy_meters)
        if cfg.enable_confidence and cfg.confidence_mode == "twod":
            currency.update_confidence_2d(head, cfg.conf_gain, cfg.conf_alpha,
                                          cfg.c_max, cfg.settled_mode, cfg.conf_k)
        # bridge into conv BEFORE the head weights move
        if self.learn_conv:
            d_feat = model.input_delta(gb)
            g = conv.backward(d_feat, cache)
            conv.update_meters(g)
            if cfg.enable_confidence:
                conv.update_confidence()
        learning.apply_gated_update(head, gw, gb, cfg.eta_base)
        for syn in head.synapses.values():
            syn.age += 1
        if self.learn_conv:
            conv.gated_update(g, self._conv_eta_now())

        self.step_idx += 1
        self._maybe_rewire(loss)
        return loss

    def _conv_eta_now(self):
        """Effective filter learning rate this step (the consolidation schedule)."""
        if not self.total_steps or self.conv_eta_schedule == "none":
            return self.conv_eta
        frac = min(self.step_idx / self.total_steps, 1.0)
        if self.conv_eta_schedule == "cosine":
            return self.conv_eta * 0.5 * (1.0 + math.cos(math.pi * frac))
        if self.conv_eta_schedule == "freeze":
            return 0.0 if frac >= self.freeze_frac else self.conv_eta
        raise ValueError(f"unknown conv_eta_schedule {self.conv_eta_schedule!r}")

    # -- phasic structure (Stage D) -----------------------------------------
    def _maybe_rewire(self, loss):
        """At a settledness plateau, rewire BOTH economies in one burst (conv
        filters first so the head can target newborn slots), then require a fresh
        plateau. Mirrors Trainer._rewire_phasic; the head reuses the exact
        currency.prune/grow primitives on a batch of CURRENT conv features."""
        cfg = self.cfg
        settled = self.detector.update(loss, self.step_idx)
        if not settled or not (cfg.enable_prune or cfg.enable_grow):
            return
        self.events.append({"step": self.step_idx, "type": "sleep"})
        if self.conv_structure:
            self._rewire_conv()
        self._rewire_head()
        self.detector.reset()

    def _rewire_conv(self):
        cfg, conv = self.cfg, self.model.conv
        if cfg.enable_prune:
            for k in conv.prune(self.conv_prune_floor, lam=cfg.lam_prune,
                                k_min=self.conv_k_min, grace=cfg.t_grace):
                self.events.append({"step": self.step_idx, "type": "conv_prune",
                                    "filter": int(k)})
            if self.conv_redundancy_prune:
                mode = self.conv_redundancy_mode
                red = []
                if mode in ("kernel", "both"):     # geometric duplicates
                    kt = self.conv_redundancy_threshold if mode == "kernel" else 0.85
                    red += conv.prune_redundant(kt, lam=cfg.lam_prune,
                                                k_min=self.conv_k_min)
                if mode in ("activation", "both"):  # functional (output) duplicates
                    b = min(self.conv_redundancy_batch, len(self.X))
                    idx = self.rng.choice(len(self.X), size=b, replace=False)
                    red += conv.prune_redundant_activation(
                        self.X[idx], self.conv_redundancy_threshold,
                        lam=cfg.lam_prune, k_min=self.conv_k_min)
                for k in red:
                    self.events.append({"step": self.step_idx,
                                        "type": "conv_redprune", "filter": int(k)})
        if cfg.enable_grow:
            for k in conv.grow(mode=self.conv_grow_mode, k_max=self.conv_k_max,
                               n=self.conv_grow_per_burst):
                self.events.append({"step": self.step_idx, "type": "conv_grow",
                                    "filter": int(k)})

    def _rewire_head(self):
        cfg, head = self.cfg, self.model.head
        if cfg.enable_prune:
            cap = (cfg.sleep_max_prune if cfg.sleep_max_prune is not None
                   else len(head.synapses))
            pruned = currency.prune_currency(
                head, cfg.t_grace, cap, cfg.sleep_prune_floor, cfg.lam_prune)
            for edge in pruned:
                self.events.append({"step": self.step_idx, "type": "prune",
                                    "edge": edge})
        if cfg.enable_grow:
            b = min(cfg.virt_batch, len(self.X))
            idx = self.rng.choice(len(self.X), size=b, replace=False)
            xfeat = np.array([self.model.conv.forward(self.X[j])[0] for j in idx])
            ghost, ref = currency.batch_edge_scores(
                head, xfeat, self.y[idx], grow_demand_k=cfg.grow_demand_k)
            grown = currency.grow_currency(head, ghost, ref, len(ghost),
                                           cfg.grow_bar_frac)
            for edge in grown:
                self.events.append({"step": self.step_idx, "type": "grow",
                                    "edge": edge})

    # -- eval ----------------------------------------------------------------
    def predict(self, X_imgs):
        return self.model.predict(X_imgs)

    def accuracy(self, X_imgs, y):
        return float((self.predict(X_imgs) == np.asarray(y)).mean())
