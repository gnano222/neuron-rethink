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


def apply_gated_update(net, grad_w, grad_b, eta_base, optimizer="sgd", eps=1e-12):
    """Confidence-gated weight update (§4.6).

        eta_eff_ij = eta_base / (1 + c_ij)
        w_ij      <- w_ij - eta_eff_ij * step_ij
        bias_j    <- bias_j - eta_base * dL/dbias_j   (ungated, always plain SGD)

    ``optimizer`` selects the step direction/scale:
      * ``"sgd"``      -> ``step = dL/dw_ij`` — the instantaneous gradient (plain
                          gated SGD; with confidence 0 this is exactly plain SGD).
      * ``"currency"`` -> ``step = S_ij / (M_ij + eps)`` — the CURRENCY-NATIVE
                          step: move along the signed gradient meter ``S``,
                          normalized by the magnitude meter ``M``. ``S/M`` is the
                          consistency coefficient ``kappa * sign(S)`` in [-1, 1]
                          (see currency.py), so it REUSES the two meters SPROUT
                          already maintains. Three properties fall out: the step is
                          self-normalized (a consistent wire moves ~eta regardless
                          of gradient scale — the adaptive-LR effect); it needs no
                          bias correction (the EMA warm-up bias cancels in the
                          ratio); and it auto-anneals (an oscillating gradient has
                          |S| << M -> step -> 0). Because steps are ~unit-scaled,
                          ``eta_base`` is the per-step weight change and wants a
                          smaller value than SGD's.

    Confident synapses barely move; unsure ones move fast.
    """
    if optimizer not in ("sgd", "currency"):
        raise ValueError(f"unknown optimizer {optimizer!r}")
    for key, syn in net.synapses.items():
        eta_eff = eta_base / (1.0 + syn.confidence)
        step = (syn.grad_signed / (syn.grad_mag + eps)
                if optimizer == "currency" else grad_w[key])
        syn.weight -= eta_eff * step

    for nid, g in grad_b.items():
        net.neurons[nid].bias -= eta_base * g
