"""Deep (stacked) Conv-SPROUT: two or more weight-shared conv layers, each a
filter-level gradient-as-currency economy, chained so features compose
(edges -> strokes -> parts). This is purely ADDITIVE — the promoted single-layer
Conv-SPROUT (sprout/conv.py ConvEconomy + conv_train.py) is untouched.

The economic unit is still the filter; the only new machinery a *stack* needs over
the single layer is (a) MULTI-CHANNEL conv (a layer-2 filter spans all the
channels its predecessor emits) and (b) INPUT-gradient backprop, so an upper
layer can pass gradient down to the layer below. Both live here and are
gradient-checked against finite differences (the project's correctness linchpin).
See docs/superpowers/specs (stacked-conv design).
"""
from __future__ import annotations

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# pooling is channel-agnostic — reuse the validated single-layer implementation
from sprout.conv import maxpool_forward, maxpool_backward
from sprout.currency import settledness

_EPS = 1e-12


# -- (A) multi-channel conv math (vectorized, gradient-checked) ---------------

def conv_mc_forward(x, kernels):
    """Valid multi-channel 2D correlation.

    ``x`` (C, H, W); ``kernels`` (K, C, kh, kw). Returns preactivations
    (K, oh, ow) with ``out[k,i,j] = sum_{c,a,b} x[c,i+a,j+b] * kernels[k,c,a,b]``.
    Reduces to the single-channel :func:`sprout.conv.conv_valid_forward` at C=1.
    """
    x = np.asarray(x, dtype=float)
    kernels = np.asarray(kernels, dtype=float)
    kh, kw = kernels.shape[2], kernels.shape[3]
    win = sliding_window_view(x, (kh, kw), axis=(1, 2))   # (C, oh, ow, kh, kw)
    return np.einsum("cijab,kcab->kij", win, kernels)


def conv_mc_filter_grad(x, d_preact):
    """Gradient of the loss w.r.t. each kernel, summed over all placements.

    ``x`` (C, H, W); ``d_preact`` (K, oh, ow). Returns g (K, C, kh, kw):
        g[k,c,a,b] = sum_{i,j} d_preact[k,i,j] * x[c, i+a, j+b]
    This placement-sum is exactly the shared-weight (conv) gradient, one channel
    axis wider than the single-channel version.
    """
    x = np.asarray(x, dtype=float)
    d_preact = np.asarray(d_preact, dtype=float)
    oh, ow = d_preact.shape[1], d_preact.shape[2]
    win = sliding_window_view(x, (oh, ow), axis=(1, 2))   # (C, kh, kw, oh, ow)
    return np.einsum("kij,cabij->kcab", d_preact, win)


def conv_mc_input_grad(d_preact, kernels):
    """Gradient of the loss w.r.t. the layer INPUT (the transposed conv) — the
    piece a single layer never needed but a stack does, to backprop downward.

    ``d_preact`` (K, oh, ow); ``kernels`` (K, C, kh, kw). Returns d_x (C, H, W)
    with H=oh+kh-1, W=ow+kw-1:
        d_x[c,p,q] = sum_{k,a,b} d_preact[k, p-a, q-b] * kernels[k,c,a,b]
    Implemented as a scatter-add over the kh*kw taps (each tap an einsum over
    K -> C); cheap because the kernel is tiny.
    """
    d_preact = np.asarray(d_preact, dtype=float)
    kernels = np.asarray(kernels, dtype=float)
    K, C, kh, kw = kernels.shape
    _, oh, ow = d_preact.shape
    d_x = np.zeros((C, oh + kh - 1, ow + kw - 1))
    for a in range(kh):
        for b in range(kw):
            d_x[:, a:a + oh, b:b + ow] += np.einsum(
                "kc,kij->cij", kernels[:, :, a, b], d_preact)
    return d_x


# -- (B) ConvLayer: a multi-channel filter-level currency economy --------------

class ConvLayer:
    """A bank of ``k`` weight-shared, multi-channel filters governed by the SAME
    gradient-as-currency the wires use, one level up: a per-filter magnitude meter
    ``M`` (EMA of the placement-summed gradient norm), a signed vector meter
    ``Svec``, a 2D confidence (importance x settledness), and the ``eta/(1+c)``
    gated update. ``forward`` returns the SPATIAL pooled maps (so layers chain);
    ``backward`` returns the filter gradient AND (optionally) the input gradient
    (to backprop into the layer below). The single-layer ``ConvEconomy`` keeps the
    self-sizing prune/grow; this stack layer is fixed-count for the depth study and
    adds those later. ``c_in=1`` is the first (image) layer.
    """

    def __init__(self, k, c_in, kh=3, kw=3, pool=2, seed=0, init_std=None,
                 beta_g=0.99, conf_gain=2.0, conf_alpha=0.01, c_max=5.0,
                 conf_k=3.0, settled_mode="sigmoid"):
        self.k, self.c_in, self.kh, self.kw, self.pool = k, c_in, kh, kw, pool
        self.beta_g = beta_g
        self.conf_gain, self.conf_alpha = conf_gain, conf_alpha
        self.c_max, self.conf_k, self.settled_mode = c_max, conf_k, settled_mode
        rng = np.random.default_rng(seed)
        std = init_std if init_std is not None else np.sqrt(2.0 / (c_in * kh * kw))
        self.theta = rng.normal(0.0, std, size=(k, c_in, kh, kw))
        self.M = np.zeros(k)
        self.Svec = np.zeros((k, c_in, kh, kw))
        self.conf = np.zeros(k)
        self.age = np.zeros(k, dtype=int)

    # -- geometry ------------------------------------------------------------
    def out_hw(self, h, w):
        return (h - self.kh + 1) // self.pool, (w - self.kw + 1) // self.pool

    def feat_dim(self, h, w):
        ph, pw = self.out_hw(h, w)
        return self.k * ph * pw

    # -- forward / backward --------------------------------------------------
    def forward(self, x):
        """``x`` (c_in, H, W) -> pooled maps (k, poh, pow) + cache."""
        preact = conv_mc_forward(x, self.theta)            # (k, oh, ow)
        relu = np.maximum(preact, 0.0)
        pooled, argmax = maxpool_forward(relu, self.pool)  # (k, poh, pow)
        return pooled, (x, preact, argmax, relu.shape)

    def backward(self, d_pooled, cache, need_input_grad=True):
        x, preact, argmax, relu_shape = cache
        d_relu = maxpool_backward(d_pooled, argmax, relu_shape, self.pool)
        d_preact = d_relu * (preact > 0)
        g = conv_mc_filter_grad(x, d_preact)               # (k, c_in, kh, kw)
        d_x = conv_mc_input_grad(d_preact, self.theta) if need_input_grad else None
        return g, d_x

    # -- currency readouts (same load / demand the wires use) ----------------
    def loads(self):
        L = np.sqrt((self.theta ** 2).sum(axis=(1, 2, 3)))
        wbar = L.mean() if self.k else 0.0
        return L / (wbar + _EPS)

    def demands(self):
        mbar = self.M.mean() if self.k else 0.0
        return self.M / (mbar + _EPS)

    # -- per-step economy updates -------------------------------------------
    def update_meters(self, g, beta_g=None):
        beta = self.beta_g if beta_g is None else beta_g
        gmag = np.sqrt((g ** 2).sum(axis=(1, 2, 3)))
        self.M = beta * self.M + (1.0 - beta) * gmag
        self.Svec = beta * self.Svec + (1.0 - beta) * g

    def update_confidence(self):
        load, dem = self.loads(), self.demands()
        for i in range(self.k):
            imp = max(load[i] - 1.0, 0.0)
            s = settledness(dem[i], self.settled_mode, self.conf_k)
            target = min(self.conf_gain * imp * s, self.c_max)
            self.conf[i] = (1.0 - self.conf_alpha) * self.conf[i] + self.conf_alpha * target

    def gated_update(self, g, eta):
        factor = eta / (1.0 + self.conf)
        self.theta -= factor[:, None, None, None] * g
        self.age += 1


# -- (C) DeepConvModel: a stack of ConvLayers feeding a sparse head ------------

class DeepConvModel:
    """Chains ``layers`` (a list of :class:`ConvLayer`, first one ``c_in=1``) then
    a sparse head Network reading the flattened maps of the LAST layer. ``backward
    _convs`` walks the chain top-down: the head input delta seeds the last layer,
    whose input gradient seeds the one below, and so on (the input-grad transposed
    conv is what makes the stack trainable). A single layer reduces to the same
    shape as ConvModel (but this path is fixed-count, for the depth study)."""

    def __init__(self, layers, head, h, w):
        self.layers = layers
        self.head = head
        self.h, self.w = h, w
        self.shapes = []                       # output (k, ph, pw) per layer
        ch, cw = h, w
        for layer in layers:
            ph, pw = layer.out_hw(ch, cw)
            self.shapes.append((layer.k, ph, pw))
            ch, cw = ph, pw
        self.last_shape = self.shapes[-1]
        feat = int(np.prod(self.last_shape))
        self.input_ids = list(head.layers[0])
        assert len(self.input_ids) == feat, (
            f"head input layer must match stack output ({len(self.input_ids)} vs {feat})")

    def feat_dim(self):
        return int(np.prod(self.last_shape))

    def forward(self, img):
        x = np.asarray(img, dtype=float).reshape(1, self.h, self.w)
        caches = []
        for layer in self.layers:
            x, cache = layer.forward(x)
            caches.append(cache)
        feat = x.reshape(-1)
        probs = self.head.forward(feat)
        return probs, feat, caches

    def input_delta(self, grad_b):
        """Loss gradient at each head INPUT neuron (= last-layer feature):
        dL/dfeat[p] = sum_post w(p,post)*delta_post. Bridges head -> conv stack."""
        head = self.head
        d = np.zeros(len(self.input_ids))
        for p, iid in enumerate(self.input_ids):
            s = 0.0
            for post in head.outgoing[iid]:
                s += head.synapses[(iid, post)].weight * grad_b.get(post, 0.0)
            d[p] = s
        return d

    def backward_convs(self, grad_b, caches):
        """Per-layer filter gradients (list, layer order). Chains downward via each
        layer's input gradient; the first (image) layer skips its input grad."""
        d_spatial = self.input_delta(grad_b).reshape(self.last_shape)
        grads = [None] * len(self.layers)
        for l in range(len(self.layers) - 1, -1, -1):
            g, d_spatial = self.layers[l].backward(d_spatial, caches[l],
                                                   need_input_grad=(l > 0))
            grads[l] = g
        return grads

    def predict(self, X_imgs):
        return np.array([self.forward(img)[0] for img in X_imgs]).argmax(axis=1)


# -- (D) DeepConvTrainer: joint wake training of the stack + head --------------

class DeepConvTrainer:
    """Trains every conv layer + the head together: one forward, head currency
    (meter -> 2D confidence -> gated update), then the head input delta is bridged
    DOWN the stack (``backward_convs``) and each conv layer runs the same currency
    updates one level up (meter -> confidence -> consolidated gated update). The
    head rewires phasically at a settledness plateau (the conv layers are
    fixed-count for the depth study). Mirrors ConvTrainer; the only new thing is
    looping the conv updates over the layer list. Filter LR consolidation (cosine)
    is shared across layers — the load-bearing stability fix from single-layer
    Conv-SPROUT."""

    def __init__(self, cfg, model, X_imgs, y, seed=0, conv_eta=None,
                 conv_eta_schedule="cosine", total_steps=None, freeze_frac=0.6):
        self.cfg = cfg
        self.model = model
        self.X = np.asarray(X_imgs, dtype=float) if X_imgs is not None else None
        self.y = np.asarray(y) if y is not None else None
        self.rng = np.random.default_rng(seed)
        self.step_idx = 0
        self.conv_eta = cfg.eta_base if conv_eta is None else conv_eta
        self.conv_eta_schedule = conv_eta_schedule
        self.total_steps = total_steps
        self.freeze_frac = freeze_frac
        self.events = []
        model.head.activation_top_k = cfg.activation_top_k
        from sprout.sleep import SettlednessDetector
        self.detector = SettlednessDetector(
            cfg.sleep_loss_beta, cfg.sleep_loss_tol, cfg.sleep_patience,
            cfg.sleep_warmup)

    def step(self, record=True):
        from sprout import currency, learning
        cfg, model = self.cfg, self.model
        head = model.head
        i = int(self.rng.integers(len(self.X)))
        img, yi = self.X[i], int(self.y[i])

        _, _, caches = model.forward(img)
        learning.update_firing_rates(head, cfg.beta)
        loss, gw, gb = head.backward(yi)
        currency.update_gradient_meters(head, gw, cfg.beta_g,
                                        step_idx=self.step_idx, lazy=cfg.lazy_meters)
        if cfg.enable_confidence and cfg.confidence_mode == "twod":
            currency.update_confidence_2d(head, cfg.conf_gain, cfg.conf_alpha,
                                          cfg.c_max, cfg.settled_mode, cfg.conf_k)
        grads = model.backward_convs(gb, caches)            # bridge head -> stack
        for layer, g in zip(model.layers, grads):
            layer.update_meters(g)
            if cfg.enable_confidence:
                layer.update_confidence()
        learning.apply_gated_update(head, gw, gb, cfg.eta_base, optimizer=cfg.optimizer)
        for syn in head.synapses.values():
            syn.age += 1
        eta = self._conv_eta_now()
        for layer, g in zip(model.layers, grads):
            layer.gated_update(g, eta)

        self.step_idx += 1
        self._maybe_rewire(loss)
        return loss

    def _conv_eta_now(self):
        if not self.total_steps or self.conv_eta_schedule == "none":
            return self.conv_eta
        frac = min(self.step_idx / self.total_steps, 1.0)
        if self.conv_eta_schedule == "cosine":
            return self.conv_eta * 0.5 * (1.0 + math.cos(math.pi * frac))
        if self.conv_eta_schedule == "freeze":
            return 0.0 if frac >= self.freeze_frac else self.conv_eta
        raise ValueError(f"unknown conv_eta_schedule {self.conv_eta_schedule!r}")

    def _maybe_rewire(self, loss):
        cfg = self.cfg
        settled = self.detector.update(loss, self.step_idx)
        if not settled or not (cfg.enable_prune or cfg.enable_grow):
            return
        self.events.append({"step": self.step_idx, "type": "sleep"})
        self._rewire_head()
        self.detector.reset()

    def _rewire_head(self):
        from sprout import currency
        cfg, head = self.cfg, self.model.head
        if cfg.enable_prune:
            cap = (cfg.sleep_max_prune if cfg.sleep_max_prune is not None
                   else len(head.synapses))
            pruned = currency.prune_currency(
                head, cfg.t_grace, cap, cfg.sleep_prune_floor, cfg.lam_prune)
            for edge in pruned:
                self.events.append({"step": self.step_idx, "type": "prune", "edge": edge})
        if cfg.enable_grow:
            b = min(cfg.virt_batch, len(self.X))
            idx = self.rng.choice(len(self.X), size=b, replace=False)
            xfeat = np.array([self.model.forward(self.X[j])[1] for j in idx])
            ghost, ref = currency.batch_edge_scores(
                head, xfeat, self.y[idx], grow_demand_k=cfg.grow_demand_k)
            grown = currency.grow_currency(head, ghost, ref, len(ghost), cfg.grow_bar_frac)
            for edge in grown:
                self.events.append({"step": self.step_idx, "type": "grow", "edge": edge})

    def predict(self, X_imgs):
        return self.model.predict(X_imgs)

    def accuracy(self, X_imgs, y):
        return float((self.predict(X_imgs) == np.asarray(y)).mean())
