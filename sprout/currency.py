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

def update_confidence_2d(net, gain, alpha, c_max, eps=_EPS):
    """Confidence as the shared 2D state prune reads: *important* AND *settled*.

        imp     = max(|w_ij|/wbar - 1, 0)     "above-average weight" (importance)
        settled = max(1 - M_ij/Mbar, 0)       "below-average demand" (settledness)
        target  = min(gain * imp * settled, c_max)
        c_ij   <- (1 - alpha) * c_ij + alpha * target     (EMA toward target)

    Unlike the tug-of-war rule this reads *weight* (the term prune utility uses),
    so confidence tracks utility instead of anti-correlating with it: a wire is
    frozen only when it carries above-average load AND the loss has stopped
    pushing it. Freeloaders (below-average weight) score ``imp = 0`` and never
    freeze; a contested wire (``M >= Mbar``) has ``settled = 0`` so its target
    collapses and confidence decays - it releases, by design. No ``m_floor`` /
    consistency gate is needed: the weight gate already excludes dead freeloaders
    while (correctly) freezing dead-but-load-bearing wires.
    """
    if not net.synapses:
        return
    wbar = sum(abs(s.weight) for s in net.synapses.values()) / len(net.synapses)
    mbar = mean_grad_mag(net)
    for syn in net.synapses.values():
        imp = max(abs(syn.weight) / (wbar + eps) - 1.0, 0.0)
        settled = max(1.0 - syn.grad_mag / (mbar + eps), 0.0)
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
    wbar = sum(abs(s.weight) for s in net.synapses.values()) / len(net.synapses)
    mbar = mean_grad_mag(net)

    candidates = []
    for (pre, post), syn in net.synapses.items():
        if syn.age <= t_grace:
            continue
        u = abs(syn.weight) / (wbar + eps) + lam * syn.grad_mag / (mbar + eps)
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
