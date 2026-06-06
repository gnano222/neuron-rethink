"""Per-step learning rules for the currency path.

  * ``update_firing_rates``  - firing-rate EMA (§4.2), kept as viz bookkeeping.
  * ``apply_gated_update``   - confidence-gated weight update (§4.6).

The legacy eligibility trace and three-factor confidence rule were removed when
the gradient-as-currency path became the only architecture (see
docs/superpowers/specs/2026-06-06-sprout-prune-grow-consolidation-design.md).
"""

from __future__ import annotations


def update_firing_rates(net, beta):
    """Firing-rate EMA, every step, all neurons (§4.2).

        r_j <- (1 - beta) * r_j + beta * a_j
    """
    for n in net.neurons:
        n.firing_rate = (1.0 - beta) * n.firing_rate + beta * n.activation


def apply_gated_update(net, grad_w, grad_b, eta_base):
    """Confidence-gated SGD step (§4.6).

        eta_eff_ij = eta_base / (1 + c_ij)
        w_ij      <- w_ij - eta_eff_ij * dL/dw_ij
        bias_j    <- bias_j - eta_base * dL/dbias_j   (ungated)

    Confident synapses barely move; unsure ones move fast. While confidence is
    0 (early training, newborn synapses) this is exactly plain SGD.
    """
    for key, syn in net.synapses.items():
        eta_eff = eta_base / (1.0 + syn.confidence)
        syn.weight -= eta_eff * grad_w[key]

    for nid, g in grad_b.items():
        net.neurons[nid].bias -= eta_base * g
