"""Structural plasticity and homeostasis (§4.7, §4.8).

  * ``homeostasis`` - gently scale incoming weights toward the target firing
    rate. Empirically this is load-bearing: the confidence / eligibility /
    growth machinery is calibrated for firing rates near ``r_target``, and
    homeostasis is what holds the network in that regime.
  * ``prune`` - remove low-utility synapses past the grace period (§4.7).
  * ``grow``  - sprout new zero-weight synapses into underfiring neurons (§4.7).
"""

from __future__ import annotations


def prune(net, theta_prune, t_grace, max_prune):
    """Remove low-utility synapses past the grace period (§4.7).

        u_ij  = |w_ij| * r_pre
        prune  if  (u_ij < theta_prune) AND (age_ij > t_grace)

    At most ``max_prune`` per round (gentle, watchable), lowest-utility first.
    A neuron's *last* incoming synapse is never pruned, so no neuron is fully
    starved of input (keeps the graph trainable; growth handles the rest).
    Returns the list of pruned ``(pre, post)`` edges.
    """
    candidates = []
    for (pre, post), syn in net.synapses.items():
        if syn.age <= t_grace:
            continue
        u = abs(syn.weight) * net.neurons[pre].firing_rate
        if u < theta_prune:
            candidates.append((u, pre, post))

    candidates.sort()  # lowest utility first
    pruned = []
    for u, pre, post in candidates:
        if len(pruned) >= max_prune:
            break
        if len(net.incoming[post]) <= 1:   # don't orphan the post neuron
            continue
        net.remove_synapse(pre, post)
        pruned.append((pre, post))
    return pruned


def grow(net, r_target, f_under, max_grow, grow_budget=None):
    """Sprout new synapses into underfiring neurons (§4.7).

    A neuron ``j`` is underfiring if ``r_j < f_under * r_target``. For the most
    underfiring neurons (capped at ``max_grow``):
      1. candidate sources ``i``: ``layer(i) < layer(j)`` and no ``i->j`` yet;
      2. pick the candidate with the highest activation in the most recent
         forward pass ("the neuron that just fired strongly before j");
      3. create ``i->j`` born at weight 0, confidence 0, eligibility 0, age 0
         (a no-op the instant it appears, born maximally plastic).
    Returns the list of new ``(pre, post)`` edges.

    ``grow_budget`` (deviation from spec): each neuron may be targeted at most
    this many times. A zero-weight synapse into a *dead* ReLU neuron gets no
    gradient and cannot revive it, so without a budget growth + pruning churn
    forever on a handful of dead units. The budget retires chronically-dead
    neurons; genuinely under-connected ones recover within a few attempts.
    """
    threshold = f_under * r_target
    underfiring = [
        (net.neurons[nid].firing_rate, nid)
        for nid in range(net.num_neurons)
        if net.neurons[nid].layer > 0 and net.neurons[nid].firing_rate < threshold
        and (grow_budget is None or net.neurons[nid].grow_attempts < grow_budget)
    ]
    underfiring.sort()  # most underfiring first

    grown = []
    for _, j in underfiring:
        if len(grown) >= max_grow:
            break
        j_layer = net.neurons[j].layer
        best, best_act = None, None
        for i in range(net.num_neurons):
            if net.neurons[i].layer >= j_layer:
                continue
            if (i, j) in net.synapses:
                continue
            act = net.neurons[i].activation
            if best is None or act > best_act:
                best, best_act = i, act
        if best is not None:
            net.add_synapse(best, j, weight=0.0, confidence=0.0, eligibility=0.0, age=0)
            net.neurons[j].grow_attempts += 1
            grown.append((best, j))
    return grown


def homeostasis(net, r_target, rho, eps=1e-6, scale_min=0.5, scale_max=1.5):
    """Push each neuron toward ``r_target`` by scaling its incoming weights (§4.8).

        scale = (r_target / (r_j + eps)) ** rho        # rho gentle (~0.1)
        w_ij <- w_ij * scale   for each incoming synapse i->j

    The per-round scale is clamped to ``[scale_min, scale_max]``. Without this,
    a dead ReLU neuron (``r_j ~ 0``) gets ``scale ~ 3.3`` which compounds into a
    runaway weight/activation explosion. NOTE: this mechanism is off by default
    - empirically the network is stable without it, and its weight-rescaling
    fights the confidence-gated learning rate, so it is kept as an opt-in.
    """
    for nid, neuron in enumerate(net.neurons):
        if neuron.layer == 0 or not net.incoming[nid]:
            continue
        scale = (r_target / (neuron.firing_rate + eps)) ** rho
        scale = min(max(scale, scale_min), scale_max)
        for pre in net.incoming[nid]:
            net.synapses[(pre, nid)].weight *= scale
