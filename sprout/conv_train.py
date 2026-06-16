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
                 conv_grow_per_burst=2):
        self.cfg = cfg
        self.model = model
        self.X = np.asarray(X_imgs, dtype=float)
        self.y = np.asarray(y)
        self.rng = np.random.default_rng(seed)
        self.step_idx = 0
        self.conv_eta = cfg.eta_base if conv_eta is None else conv_eta
        self.learn_conv = learn_conv
        self.conv_structure = conv_structure
        self.conv_k_max = model.conv.k_max if conv_k_max is None else conv_k_max
        self.conv_grow_mode = conv_grow_mode
        self.conv_prune_floor = conv_prune_floor
        self.conv_k_min = conv_k_min
        self.conv_grow_per_burst = conv_grow_per_burst
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
            conv.gated_update(g, self.conv_eta)

        self.step_idx += 1
        self._maybe_rewire(loss)
        return loss

    # filled in Stage D; here it only advances the settledness detector
    def _maybe_rewire(self, loss):
        self.detector.update(loss, self.step_idx)

    # -- eval ----------------------------------------------------------------
    def predict(self, X_imgs):
        return self.model.predict(X_imgs)

    def accuracy(self, X_imgs, y):
        return float((self.predict(X_imgs) == np.asarray(y)).mean())
