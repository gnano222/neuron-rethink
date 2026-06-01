"""Gradient-as-currency mechanism (enhancement on top of the v1 spec).

The idea: backprop already computes, for every wire, a per-step gradient
``g_ij = dL/dw_ij = delta_j * a_i`` - "how much, and which way, the loss wants
this wire to change". We meter that single quantity once per step and then read
it three different ways:

  * Readout A - confidence  (existing wires): calm + consistent => consolidate.
  * Readout B - pruning     (existing wires): inert (small weight AND ignored).
  * Readout C - growth      (missing wires): grow the ghost wire the loss most
                wishes existed (RigL-style virtual gradient).

This replaces the v1 eligibility trace + global exp(-loss) confidence + activity
based growth + |w|*r pruning with one coherent signal. See currency.md / README
for the formulation and the honest trade-offs (notably: less biologically local
than the Hebbian eligibility it replaces).

Notation matches the design notes:
    M_ij = grad_mag    (EMA of |g_ij|)            "how hard am I being pushed"
    S_ij = grad_signed (EMA of  g_ij)             "which way, on net"
    d_ij = M_ij / mean(M)                          "demand vs the network"
    kappa_ij = |S_ij| / (M_ij + eps)  in [0,1]     "is the feedback consistent"
"""

from __future__ import annotations

import math
from collections import defaultdict

_EPS = 1e-12


# -- §1 the currency: per-wire gradient meters -------------------------------

def update_gradient_meters(net, grad_w, beta):
    """EMA-update each live wire's magnitude and signed gradient meters.

        M_ij <- beta * M_ij + (1 - beta) * |g_ij|
        S_ij <- beta * S_ij + (1 - beta) *  g_ij
    """
    for key, syn in net.synapses.items():
        g = grad_w.get(key, 0.0)
        syn.grad_mag = beta * syn.grad_mag + (1.0 - beta) * abs(g)
        syn.grad_signed = beta * syn.grad_signed + (1.0 - beta) * g


def mean_grad_mag(net):
    """Mean magnitude meter over live wires (the adaptive scale ``M-bar``)."""
    if not net.synapses:
        return 0.0
    return sum(s.grad_mag for s in net.synapses.values()) / len(net.synapses)


# -- shared state: the two coordinates both lenses read ----------------------

def network_scales(net):
    """The two adaptive scales every wire is normalised against:

        (w-bar = mean |weight|,  M-bar = mean grad_mag)  over live wires.

    Both the confidence (plasticity) lens and the prune (value) lens read the
    wire's load and demand relative to these, so they share one definition of
    "the network" rather than each recomputing its own.
    """
    n = len(net.synapses)
    if n == 0:
        return 0.0, 0.0
    wbar = sum(abs(s.weight) for s in net.synapses.values()) / n
    return wbar, mean_grad_mag(net)


def load(syn, wbar, eps=_EPS):
    """``ℓ = |w| / w-bar`` — the wire's weight relative to the network average."""
    return abs(syn.weight) / (wbar + eps)


def demand(syn, Mbar, eps=_EPS):
    """``d = M / M-bar`` — how hard the loss still pushes this wire vs average."""
    return syn.grad_mag / (Mbar + eps)


def _sigmoid(z):
    """Numerically stable logistic (no overflow for large |z|)."""
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def settledness(d, mode="sigmoid", k=3.0):
    """Map relative demand ``d = M/M-bar`` to settledness in (0, 1]: high when the
    loss has stopped pushing this wire.

      * ``hard``     -> ``max(1 - d, 0)``         the original ReLU cliff: slams to
                        exactly zero for *every* wire with above-average demand.
      * ``sigmoid``  -> ``σ(k·(1 - d))``          smooth version of the cliff; still
                        pivots at average demand (``d=1`` -> 0.5).
      * ``exp``      -> ``exp(-k·d)``             decays from the quietest wire.
      * ``rational`` -> ``1 / (1 + k·d)``         fattest tail (slowest decay).

    The smooth modes are strictly positive, so a *load-bearing* wire that is
    briefly contested keeps a small, weight-proportional consolidation instead of
    collapsing to zero confidence — lifting the high-demand tail off the axis.
    """
    if mode == "hard":
        return max(1.0 - d, 0.0)
    if mode == "sigmoid":
        return _sigmoid(k * (1.0 - d))
    if mode == "exp":
        return math.exp(-k * d)
    if mode == "rational":
        return 1.0 / (1.0 + k * d)
    raise ValueError(f"unknown settled_mode {mode!r}")


# -- Readout A: confidence ---------------------------------------------------

def update_confidence_currency(net, gamma_dec, gamma_up, gamma_dn, c_max,
                               m_floor_frac, eps=_EPS):
    """Confidence as a tug-of-war on the currency (every step, all live wires).

        d     = M_ij / (M-bar + eps)
        kappa = |S_ij| / (M_ij + eps)   if M_ij >= m_floor else 0
        c_ij <- clip( (1 - gamma_dec) * c_ij
                       + gamma_up * kappa * max(1 - d, 0)      # earn (calm+consistent)
                       - gamma_dn * max(d - 1, 0),             # lose (contested hot-spot)
                      0, c_max )

    The ``kappa`` factor is what stops a *dead* wire (no feedback, hence
    trivially "calm") from accruing confidence: with M_ij below the floor we set
    kappa = 0, so the earn term vanishes. ``gamma_dn > gamma_up`` makes
    confidence hard to earn and easy to lose - the desired "trust" dynamics.
    """
    mbar = mean_grad_mag(net)
    m_floor = m_floor_frac * mbar
    for syn in net.synapses.values():
        M, S = syn.grad_mag, syn.grad_signed
        d = M / (mbar + eps)
        kappa = (abs(S) / (M + eps)) if M >= m_floor else 0.0
        earn = gamma_up * kappa * max(1.0 - d, 0.0)
        lose = gamma_dn * max(d - 1.0, 0.0)
        c = (1.0 - gamma_dec) * syn.confidence + earn - lose
        syn.confidence = min(max(c, 0.0), c_max)


# -- Readout A (2D): confidence from importance x settledness ----------------

def update_confidence_2d(net, gain, alpha, c_max, settled_mode="sigmoid",
                         conf_k=3.0, eps=_EPS):
    """Confidence as the shared 2D state prune reads: *important* AND *settled*.

        imp     = max(load(syn) - 1, 0)            "above-average weight" (importance)
        settled = settledness(demand(syn), mode)   "the loss has stopped pushing"
        target  = min(gain * imp * settled, c_max)
        c_ij   <- (1 - alpha) * c_ij + alpha * target     (EMA toward target)

    Reads the *same* ``load``/``demand`` coordinates the prune lens reads (one
    shared state), so confidence tracks utility instead of anti-correlating with
    it: a wire is frozen only when it carries above-average load AND the loss has
    stopped pushing it. Freeloaders (below-average weight) score ``imp = 0`` and
    never freeze — the ``imp`` floor is deliberately kept hard.

    ``settled`` is pluggable (``settled_mode``): the original hard cliff slams a
    contested wire's confidence to zero the instant demand crosses the average;
    the softened cliffs (sigmoid/exp/rational) leave a load-bearing-but-contested
    wire a small, weight-proportional consolidation instead, so confidence varies
    smoothly with demand rather than dropping off a corner.
    """
    if not net.synapses:
        return
    wbar, mbar = network_scales(net)
    for syn in net.synapses.values():
        imp = max(load(syn, wbar, eps) - 1.0, 0.0)
        settled = settledness(demand(syn, mbar, eps), settled_mode, conf_k)
        target = min(gain * imp * settled, c_max)
        c = (1.0 - alpha) * syn.confidence + alpha * target
        syn.confidence = min(max(c, 0.0), c_max)


# -- Readout B: pruning ------------------------------------------------------

def prune_currency(net, t_grace, max_prune, prune_u_floor=0.5, lam=1.0, eps=1e-9):
    """Prune wires the network has genuinely stopped using.

        U_ij = |w_ij| / (w-bar + eps)  +  lam * M_ij / (M-bar + eps)
        prune if  U_ij < prune_u_floor  AND  age_ij > t_grace

    Both terms are normalised, so an *average* wire scores ~ (1 + lam) and the
    floor (~0.5) catches only wires that are weak in BOTH senses: small weight
    and ignored by the loss. A small-weight-but-actively-wanted wire (a newborn
    finding its feet) has a large gradient term and is protected - which is why
    this needs no separate ``prune_warmup``. Grace period + orphan guard as v1.
    Returns the list of pruned ``(pre, post)`` edges, lowest utility first.
    """
    if not net.synapses:
        return []
    wbar, mbar = network_scales(net)               # same shared state confidence reads

    candidates = []
    for (pre, post), syn in net.synapses.items():
        if syn.age <= t_grace:
            continue
        u = load(syn, wbar, eps) + lam * demand(syn, mbar, eps)
        if u < prune_u_floor:
            candidates.append((u, pre, post))

    candidates.sort()
    pruned = []
    for _, pre, post in candidates:
        if len(pruned) >= max_prune:
            break
        if len(net.incoming[post]) <= 1:     # never orphan the post neuron
            continue
        net.remove_synapse(pre, post)
        pruned.append((pre, post))
    return pruned


# -- Readout C: growth (RigL-style virtual gradient) -------------------------

def batch_edge_scores(net, X, y):
    """Score every *candidate* (missing) wire by its virtual gradient over a
    small batch, plus a reference scale from the live wires.

        g_ij^virt = delta_j * a_i      for (i,j) not an edge, layer(i) < layer(j)
        score_ij  = mean over the batch of |g_ij^virt|

    ``delta_j`` comes from backprop (``grad_b``) and ``a_i`` from the forward
    pass, so a wire's would-be usefulness is read off without building it. A
    dead neuron has ``delta_j = 0`` (no gradient flows), so ghosts into it score
    ~0 and are never grown - the system stops wasting growth on corpses for free.

    Returns ``(ghost_scores, ref)`` where ``ref`` is the mean |gradient| over
    live wires (the calibration scale for the growth bar).
    """
    ghost = defaultdict(float)
    live_sum, live_n = 0.0, 0
    n = len(X)
    for xi, yi in zip(X, y):
        net.forward(xi)
        _, grad_w, grad_b = net.backward(int(yi))
        for g in grad_w.values():
            live_sum += abs(g)
            live_n += 1
        for j in range(net.num_neurons):
            lj = net.neurons[j].layer
            if lj == 0:
                continue
            dj = grad_b.get(j, 0.0)
            if dj == 0.0:
                continue
            for i in range(net.num_neurons):
                if net.neurons[i].layer >= lj:
                    continue
                if (i, j) in net.synapses:
                    continue
                ghost[(i, j)] += abs(dj * net.neurons[i].activation)

    ghost = {k: v / n for k, v in ghost.items()}
    ref = (live_sum / live_n) if live_n else 0.0
    return ghost, ref


def update_ghost_meter(meter, ghost_scores, beta, drop_floor=1e-9):
    """EMA-update the persistent *ghost-gradient meter* across rewire cycles (A2).

        meter[e] <- beta * meter[e] + (1 - beta) * score_e   (score 0 if unseen)

    ``meter`` is a dict ``{(pre, post) -> ema}`` mutated in place. ``ghost_scores``
    is this cycle's instantaneous virtual-gradient batch (from
    :func:`batch_edge_scores`). The point is to grow on a *sustained* signal, not
    one noisy batch:

      * A brand-new ghost enters at only ``(1-beta)*score`` and must persist over
        several cycles before its EMA can clear the grow bar — so a wire that was
        just pruned (and therefore has no meter entry) cannot be re-requested on
        the very next cycle's spike. That lag *is* the anti-oscillation refractory.
      * An entry not seen this cycle decays by ``beta`` and is dropped once it
        falls below ``drop_floor``, keeping the dict bounded to currently-wanted
        candidates.

    Higher ``beta`` => longer memory => stronger refractory (and slower to grow a
    newly-needed wire). Returns ``meter`` for convenience.
    """
    for e in list(meter):                       # decay existing (add score if seen)
        m = beta * meter[e] + (1.0 - beta) * ghost_scores.get(e, 0.0)
        if m < drop_floor:
            del meter[e]
        else:
            meter[e] = m
    for e, s in ghost_scores.items():           # brand-new ghosts enter from zero
        if e not in meter:
            m = (1.0 - beta) * s
            if m >= drop_floor:
                meter[e] = m
    return meter


def grow_currency(net, ghost_scores, ref, max_grow, grow_bar_frac):
    """Grow the most-wanted ghost wires, born at weight 0.

    A ghost is grown only if its virtual gradient exceeds ``grow_bar_frac * ref``
    - i.e. the loss wants it *more than it is currently pushing a typical live
    wire*. The relative bar means growth slows to a stop once nothing is strongly
    wanted, giving the legible "grow then stabilize" curve without a fixed
    budget. Returns the list of new ``(pre, post)`` edges.
    """
    bar = grow_bar_frac * ref
    ranked = sorted(((s, k) for k, s in ghost_scores.items() if s > bar),
                    reverse=True)
    grown = []
    for _, (i, j) in ranked:
        if len(grown) >= max_grow:
            break
        net.add_synapse(i, j, weight=0.0)     # born plastic, no disruption
        grown.append((i, j))
    return grown
