"""Vectorized (array) backend for SPROUT — Approach 1: a flat edge-list with
scatter / segment-sums in pure NumPy. Faithful to the object ``Network`` (the
reference): the per-step WAKE path is vectorized; the rare phasic prune/grow
bursts round-trip to the object model and reuse the tested ``currency`` code.

See docs/superpowers/specs/2026-06-15-vectorized-backend-design.md. Single-sample
(identical online dynamics) — the speedup is from vectorizing across edges, not
from minibatching.
"""

from __future__ import annotations

import numpy as np

from sprout.network import _softmax

_EPS = 1e-12   # matches currency._EPS used in load()/demand()


def _sigmoid_vec(z):
    """Numerically stable logistic, elementwise (mirrors currency._sigmoid)."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _settledness_vec(d, mode="sigmoid", k=3.0):
    """Vectorized ``currency.settledness`` (elementwise on a demand array)."""
    if mode == "hard":
        return np.maximum(1.0 - d, 0.0)
    if mode == "sigmoid":
        return _sigmoid_vec(k * (1.0 - d))
    if mode == "exp":
        return np.exp(-k * d)
    if mode == "rational":
        return 1.0 / (1.0 + k * d)
    raise ValueError(f"unknown settled_mode {mode!r}")


class ArrayNet:
    """Array view of a :class:`sprout.network.Network` with a vectorized wake step."""

    def __init__(self, net):
        self._build(net)

    @classmethod
    def from_network(cls, net):
        return cls(net)

    # -- representation -----------------------------------------------------
    def _build(self, net):
        self.N = net.num_neurons
        self.n_layers = len(net.layers)
        self.last = self.n_layers - 1
        self.bias = np.array([n.bias for n in net.neurons], dtype=float)
        self.layer = np.array([n.layer for n in net.neurons], dtype=int)
        self.firing_rate = np.array([n.firing_rate for n in net.neurons], dtype=float)
        self.input_ids = np.array(net.layers[0], dtype=int)
        self.layer_neurons = [np.array(L, dtype=int) for L in net.layers]

        self.keys = list(net.synapses.keys())          # stable edge order
        self.E = len(self.keys)
        syns = [net.synapses[k] for k in self.keys]
        self.pre = np.array([k[0] for k in self.keys], dtype=int) if self.E else np.zeros(0, int)
        self.post = np.array([k[1] for k in self.keys], dtype=int) if self.E else np.zeros(0, int)
        self.weight = np.array([s.weight for s in syns], dtype=float)
        self.confidence = np.array([s.confidence for s in syns], dtype=float)
        self.age = np.array([s.age for s in syns], dtype=int)
        self.grad_mag = np.array([s.grad_mag for s in syns], dtype=float)
        self.grad_signed = np.array([s.grad_signed for s in syns], dtype=float)
        self.grad_last_step = np.array([s.grad_last_step for s in syns], dtype=int)

        post_layer = self.layer[self.post] if self.E else np.zeros(0, int)
        pre_layer = self.layer[self.pre] if self.E else np.zeros(0, int)
        self.in_edges = [np.where(post_layer == L)[0] for L in range(self.n_layers)]
        self.out_edges = [np.where(pre_layer == L)[0] for L in range(self.n_layers)]

        self.activation = np.zeros(self.N, dtype=float)
        self.delta = np.zeros(self.N, dtype=float)
        self.activation_top_k = getattr(net, "activation_top_k", None)

    def sync_into(self, net):
        """Write array state back into the object Network (used at rare bursts)."""
        for i, n in enumerate(net.neurons):
            n.bias = float(self.bias[i])
            n.firing_rate = float(self.firing_rate[i])
        for r, k in enumerate(self.keys):
            s = net.synapses[k]
            s.weight = float(self.weight[r])
            s.confidence = float(self.confidence[r])
            s.age = int(self.age[r])
            s.grad_mag = float(self.grad_mag[r])
            s.grad_signed = float(self.grad_signed[r])
            s.grad_last_step = int(self.grad_last_step[r])

    # -- forward / backward -------------------------------------------------
    def _layer_preact(self, ids, edges):
        """z for one layer = bias + scatter-add of weight*a_pre over its edges."""
        z = self.bias[ids].copy()
        if edges.size:
            contrib = self.weight[edges] * self.activation[self.pre[edges]]
            acc = np.bincount(self.post[edges], weights=contrib, minlength=self.N)
            z = z + acc[ids]
        return z

    def forward(self, x):
        a = self.activation
        a[:] = 0.0
        a[self.input_ids] = np.asarray(x, dtype=float)
        top_k = self.activation_top_k
        for L in range(1, self.last):
            ids = self.layer_neurons[L]
            zL = self._layer_preact(ids, self.in_edges[L])
            aL = np.where(zL > 0.0, zL, 0.0)
            if top_k is not None and top_k >= 0:
                posloc = np.where(aL > 0.0)[0]
                if posloc.size > top_k:                # keep top-k positives
                    order = np.lexsort((ids[posloc], aL[posloc]))[::-1]
                    keep = np.zeros(ids.size, dtype=bool)
                    keep[posloc[order[:top_k]]] = True
                    aL = np.where(keep, aL, 0.0)
            a[ids] = aL
        out = self.layer_neurons[self.last]
        zO = self._layer_preact(out, self.in_edges[self.last])
        probs = _softmax(zO)
        a[out] = probs
        return probs

    def backward(self, y_true):
        a = self.activation
        out = self.layer_neurons[self.last]
        probs = a[out]
        loss = float(-np.log(probs[y_true] + 1e-12))
        d = self.delta
        d[:] = 0.0
        d[out] = probs
        d[out[y_true]] -= 1.0
        for L in range(self.last - 1, 0, -1):
            ids = self.layer_neurons[L]
            e = self.out_edges[L]
            if e.size:
                contrib = self.weight[e] * d[self.post[e]]
                acc = np.bincount(self.pre[e], weights=contrib, minlength=self.N)
                accL = acc[ids]
            else:
                accL = np.zeros(ids.size)
            d[ids] = accL * (a[ids] > 0.0)
        grad_w = d[self.post] * a[self.pre] if self.E else np.zeros(0)
        return loss, grad_w

    # -- per-step wake update (mirrors Trainer._step_currency order) --------
    def step(self, x, y_true, cfg, step_idx):
        probs = self.forward(x)
        self.firing_rate = (1.0 - cfg.beta) * self.firing_rate + cfg.beta * self.activation
        loss, grad_w = self.backward(int(y_true))

        bg = cfg.beta_g                                  # §1 currency meters (non-lazy)
        self.grad_mag = bg * self.grad_mag + (1.0 - bg) * np.abs(grad_w)
        self.grad_signed = bg * self.grad_signed + (1.0 - bg) * grad_w
        if self.E:
            self.grad_last_step[:] = step_idx + 1

        if cfg.enable_confidence:                        # Readout A
            if cfg.confidence_mode == "twod":
                self._update_confidence_2d(cfg)
            else:
                raise NotImplementedError(
                    "ArrayNet supports confidence_mode='twod' only; "
                    f"got {cfg.confidence_mode!r}")

        if self.E:                                       # gated SGD
            self.weight -= (cfg.eta_base / (1.0 + self.confidence)) * grad_w
        noninput = self.layer > 0
        self.bias[noninput] -= cfg.eta_base * self.delta[noninput]
        self.age += 1
        return loss

    def _update_confidence_2d(self, cfg):
        if not self.E:
            return
        wbar = float(np.mean(np.abs(self.weight)))
        mbar = float(np.mean(self.grad_mag))
        load = np.abs(self.weight) / (wbar + _EPS)
        imp = np.maximum(load - 1.0, 0.0)
        demand = self.grad_mag / (mbar + _EPS)
        settled = _settledness_vec(demand, cfg.settled_mode, cfg.conf_k)
        target = np.minimum(cfg.conf_gain * imp * settled, cfg.c_max)
        c = (1.0 - cfg.conf_alpha) * self.confidence + cfg.conf_alpha * target
        self.confidence = np.clip(c, 0.0, cfg.c_max)


# -- ArrayTrainer: a Trainer-shaped adapter the eval harness can drive ---------

class ArrayTrainer:
    """Mirrors the slice of ``Trainer`` that ``evals.runner.run_one`` uses, but
    trains on the vectorized ``ArrayNet`` wake path. Rare phasic rewire bursts
    round-trip to the object ``net`` and reuse the tested ``currency`` prune/grow.

    Supported: phasic (``phasic_structure=True``) and static (no plasticity).
    Continuous-with-plasticity (``phasic_structure=False`` + prune/grow) raises —
    those A/B variants stay on the object backend.
    """

    def __init__(self, cfg, net, X, y, seed=0):
        import math
        from sprout.sleep import SettlednessDetector
        if not cfg.phasic_structure and (cfg.enable_prune or cfg.enable_grow):
            raise NotImplementedError(
                "ArrayTrainer supports phasic_structure=True or no-plasticity; "
                "use backend='object' for continuous-path prune/grow.")
        self.cfg = cfg
        self.net = net
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y)
        self.rng = np.random.default_rng(seed)
        self.an = ArrayNet.from_network(net)
        self.step_idx = 0
        self.events = []
        floor = (cfg.startle_floor if cfg.startle_floor is not None
                 else 0.5 * math.log(len(net.layers[-1])))
        self.detector = SettlednessDetector(
            cfg.sleep_loss_beta, cfg.sleep_loss_tol, cfg.sleep_patience,
            cfg.sleep_warmup, spike_tol=cfg.startle_tol,
            spike_patience=cfg.startle_patience, spike_floor=floor)

    def sync_into(self, net):
        self.an.sync_into(net)

    def step(self, record=False):
        cfg = self.cfg
        i = self.rng.integers(len(self.X))
        loss = self.an.step(self.X[i], int(self.y[i]), cfg, self.step_idx)
        settled = self.detector.update(loss, self.step_idx)
        if cfg.phasic_structure and settled and (cfg.enable_prune or cfg.enable_grow):
            self._burst()
        self.step_idx += 1
        return loss

    def _burst(self):
        from sprout.currency import (prune_currency, grow_currency,
                                     batch_edge_scores)
        cfg, net = self.cfg, self.net
        self.an.sync_into(net)                            # to the reference
        self.events.append({"step": self.step_idx, "type": "sleep", "edge": None})
        if cfg.enable_prune:
            cap = (cfg.sleep_max_prune if cfg.sleep_max_prune is not None
                   else len(net.synapses))
            for edge in prune_currency(net, cfg.t_grace, cap,
                                       cfg.sleep_prune_floor, cfg.lam_prune):
                self.events.append({"step": self.step_idx, "type": "prune", "edge": edge})
        if cfg.enable_grow:
            b = min(cfg.virt_batch, len(self.X))
            idx = self.rng.choice(len(self.X), size=b, replace=False)
            ghost, ref = batch_edge_scores(net, self.X[idx], self.y[idx],
                                           grow_demand_k=cfg.grow_demand_k)
            for edge in grow_currency(net, ghost, ref, len(ghost), cfg.grow_bar_frac):
                self.events.append({"step": self.step_idx, "type": "grow", "edge": edge})
        self.an = ArrayNet.from_network(net)             # rebuild arrays + groups
        self.detector.reset()


# -- standalone runner (thin wrapper over ArrayTrainer) -----------------------

def train_array(cfg, net, X, y, seed, steps):
    """Train ``net`` on the fast path and return it (mutated). Vectorized wake
    steps + object-reuse phasic bursts, via :class:`ArrayTrainer`."""
    tr = ArrayTrainer(cfg, net, X, y, seed=seed)
    for _ in range(steps):
        tr.step()
    tr.sync_into(net)
    return net
