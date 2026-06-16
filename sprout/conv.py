"""Conv-SPROUT (Phase 2): a weight-shared convolutional layer governed by the
gradient-as-currency economy.

Phase 1 (sprout/conv_features.py) showed FIXED translation-invariant features beat
raw pixels. Phase 2 asks whether the economy can DISCOVER better filters than a
fixed bank while staying true to SPROUT. The economic unit moves up one level from
the wire to the **filter**: a filter's taps are its weight-shared parameters,
applying it densely across the image is its placements, and the gradient meter on
the filter aggregates the gradient over every placement (== the conv weight
gradient). Confidence / prune / grow are the SAME currency readouts, one level up.
See docs/superpowers/specs/2026-06-16-conv-sprout-phase2-design.md.

This module holds (A) the pure, vectorized, gradient-checked conv/pool math and
(B) the ``ConvEconomy`` layer. The joint trainer lives in sprout/conv_train.py.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from sprout.currency import settledness

_EPS = 1e-12


# -- (A) pure conv / pool forward + backward (vectorized) ---------------------

def conv_valid_forward(img, kernels):
    """Valid 2D correlation of one image with K kernels.

    ``img`` (H,W); ``kernels`` (K,kh,kw). Returns preactivations (K,oh,ow) with
    oh=H-kh+1, ow=W-kw+1. (Correlation, not convolution -- we learn features, so
    no kernel flip; the gradient is self-consistent.)
    """
    img = np.asarray(img, dtype=float)
    kernels = np.asarray(kernels, dtype=float)
    kh, kw = kernels.shape[1], kernels.shape[2]
    win = sliding_window_view(img, (kh, kw))           # (oh,ow,kh,kw)
    return np.einsum("ijab,kab->kij", win, kernels)


def conv_valid_filter_grad(img, d_preact):
    """Gradient of the loss w.r.t. each kernel, summed over all placements.

    ``img`` (H,W); ``d_preact`` (K,oh,ow). Returns g (K,kh,kw):
        g[k,a,b] = sum_{i,j} d_preact[k,i,j] * img[i+a, j+b]
    This placement-sum is exactly the shared-weight (conv) gradient.
    """
    img = np.asarray(img, dtype=float)
    d_preact = np.asarray(d_preact, dtype=float)
    oh, ow = d_preact.shape[1], d_preact.shape[2]
    win = sliding_window_view(img, (oh, ow))           # (kh,kw,oh,ow)
    return np.einsum("kij,abij->kab", d_preact, win)


def maxpool_forward(a, p):
    """Non-overlapping p x p max-pool over the last two axes of ``a`` (K,oh,ow).

    Returns ``(pooled (K,poh,pow), argmax (K,poh,pow))`` where argmax is the flat
    index in [0, p*p) of the winning cell within each pool window (for backward).
    A ragged edge (oh or ow not divisible by p) is dropped.
    """
    a = np.asarray(a, dtype=float)
    k, oh, ow = a.shape
    poh, pow_ = oh // p, ow // p
    a = a[:, :poh * p, :pow_ * p].reshape(k, poh, p, pow_, p)
    a = a.transpose(0, 1, 3, 2, 4).reshape(k, poh, pow_, p * p)
    argmax = a.argmax(axis=3)
    pooled = np.take_along_axis(a, argmax[..., None], axis=3)[..., 0]
    return pooled, argmax


def maxpool_backward(d_pooled, argmax, a_shape, p):
    """Route ``d_pooled`` (K,poh,pow) back to the argmax cell of each window.

    Returns d_a of shape ``a_shape`` (K,oh,ow); zeros on non-max cells and on any
    dropped ragged edge.
    """
    k, oh, ow = a_shape
    poh, pow_ = oh // p, ow // p
    flat = np.zeros((k, poh, pow_, p * p))
    np.put_along_axis(flat, argmax[..., None], d_pooled[..., None], axis=3)
    flat = flat.reshape(k, poh, pow_, p, p).transpose(0, 1, 3, 2, 4)
    d_trim = flat.reshape(k, poh * p, pow_ * p)
    d_a = np.zeros(a_shape)
    d_a[:, :poh * p, :pow_ * p] = d_trim
    return d_a


# -- (B) ConvEconomy: the filter-level gradient-as-currency layer --------------

class ConvEconomy:
    """A bank of weight-shared filters governed by gradient-as-currency.

    ``k_max`` slots; ``k_init`` start active, the rest are zeroed/inactive (a
    zero kernel produces zero preactivation -> zero feature AND zero gradient, so
    an empty slot is naturally inert and occupies a fixed head-input position).
    Keeping head-input dim FIXED at ``k_max * poh * pow`` is what lets filters be
    born and pruned without ever resizing the head. Each filter carries the same
    currency state a wire does, one level up: a magnitude meter ``M`` (EMA of the
    placement-summed gradient norm), a signed vector meter ``Svec``, a 2D
    confidence, and an age. See the module docstring.
    """

    def __init__(self, k_max, kh, kw, k_init=None, pool=2, seed=0,
                 kernels=None, init_std=None, beta_g=0.99,
                 conf_gain=2.0, conf_alpha=0.01, c_max=5.0, conf_k=3.0,
                 settled_mode="sigmoid", prune_grace=200):
        self.k_max, self.kh, self.kw, self.pool = k_max, kh, kw, pool
        self.beta_g = beta_g
        self.conf_gain, self.conf_alpha = conf_gain, conf_alpha
        self.c_max, self.conf_k, self.settled_mode = c_max, conf_k, settled_mode
        self.prune_grace = prune_grace
        self._rng = np.random.default_rng(seed)

        self.theta = np.zeros((k_max, kh, kw))
        self.active = np.zeros(k_max, dtype=bool)
        std = init_std if init_std is not None else np.sqrt(2.0 / (kh * kw))
        if kernels is not None:
            kernels = np.asarray(kernels, dtype=float)
            n = len(kernels)
            self.theta[:n] = kernels
            self.active[:n] = True
        else:
            k_init = k_max if k_init is None else k_init
            self.theta[:k_init] = self._rng.normal(0.0, std, size=(k_init, kh, kw))
            self.active[:k_init] = True
        self._init_std = std

        self.M = np.zeros(k_max)
        self.Svec = np.zeros((k_max, kh, kw))
        self.conf = np.zeros(k_max)
        self.age = np.zeros(k_max, dtype=int)

    # -- geometry ------------------------------------------------------------
    @property
    def n_active(self) -> int:
        return int(self.active.sum())

    def feat_dim(self, h, w) -> int:
        oh, ow = h - self.kh + 1, w - self.kw + 1
        return self.k_max * (oh // self.pool) * (ow // self.pool)

    # -- forward / backward --------------------------------------------------
    def forward(self, img):
        preact = conv_valid_forward(img, self.theta)        # (k_max, oh, ow)
        relu = np.maximum(preact, 0.0)
        pooled, argmax = maxpool_forward(relu, self.pool)   # (k_max, poh, pow)
        return pooled.reshape(-1), (img, preact, argmax, relu.shape)

    def backward(self, d_feat, cache):
        img, preact, argmax, relu_shape = cache
        k, oh, ow = relu_shape
        poh, pow_ = oh // self.pool, ow // self.pool
        d_pooled = np.asarray(d_feat, dtype=float).reshape(k, poh, pow_)
        d_relu = maxpool_backward(d_pooled, argmax, relu_shape, self.pool)
        d_preact = d_relu * (preact > 0)
        return conv_valid_filter_grad(img, d_preact)        # (k_max, kh, kw)

    # -- currency readouts (the same load / demand the wires use) ------------
    def loads(self):
        L = np.sqrt((self.theta ** 2).sum(axis=(1, 2)))
        wbar = L[self.active].mean() if self.active.any() else 0.0
        return L / (wbar + _EPS)

    def demands(self):
        mbar = self.M[self.active].mean() if self.active.any() else 0.0
        return self.M / (mbar + _EPS)

    # -- per-step economy updates -------------------------------------------
    def update_meters(self, g, beta_g=None):
        beta = self.beta_g if beta_g is None else beta_g
        gmag = np.sqrt((g ** 2).sum(axis=(1, 2)))
        self.M = beta * self.M + (1.0 - beta) * gmag
        self.Svec = beta * self.Svec + (1.0 - beta) * g

    def update_confidence(self):
        load, dem = self.loads(), self.demands()
        for k in range(self.k_max):
            if not self.active[k]:
                self.conf[k] = 0.0
                continue
            imp = max(load[k] - 1.0, 0.0)
            s = settledness(dem[k], self.settled_mode, self.conf_k)
            target = min(self.conf_gain * imp * s, self.c_max)
            self.conf[k] = (1.0 - self.conf_alpha) * self.conf[k] + self.conf_alpha * target

    def gated_update(self, g, eta):
        factor = eta / (1.0 + self.conf)
        self.theta -= factor[:, None, None] * g
        self.age[self.active] += 1

    # -- phasic structure (filter birth / death) ----------------------------
    def prune(self, floor, lam=1.0, k_min=1, grace=None):
        grace = self.prune_grace if grace is None else grace
        load, dem = self.loads(), self.demands()
        cand = sorted((load[k] + lam * dem[k], k) for k in range(self.k_max)
                      if self.active[k] and self.age[k] > grace
                      and load[k] + lam * dem[k] < floor)
        pruned = []
        for _, k in cand:
            if self.n_active - len(pruned) <= k_min:
                break
            self.active[k] = False
            self.theta[k] = 0.0
            self.M[k] = 0.0
            self.Svec[k] = 0.0
            self.conf[k] = 0.0
            pruned.append(k)
        return pruned

    def grow(self, mode="split", k_max=None, n=1):
        cap = self.k_max if k_max is None else k_max
        born = []
        for _ in range(n):
            if self.n_active >= cap:
                break
            inactive = np.where(~self.active)[0]
            if len(inactive) == 0:
                break
            slot = int(inactive[0])
            if mode == "random":
                self.theta[slot] = self._rng.normal(
                    0.0, self._init_std, size=(self.kh, self.kw))
            elif mode == "split":
                dem = self.demands()
                act_idx = np.where(self.active)[0]
                parent = int(act_idx[np.argmax(dem[act_idx])])
                self.theta[slot] = self.theta[parent] + self._rng.normal(
                    0.0, 1e-3, size=(self.kh, self.kw))
            else:
                raise ValueError(f"unknown grow mode {mode!r}")
            self.active[slot] = True
            self.M[slot] = 0.0
            self.Svec[slot] = 0.0
            self.conf[slot] = 0.0
            self.age[slot] = 0
            born.append(slot)
        return born
