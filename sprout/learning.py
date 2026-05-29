"""Per-step learning rules (§4.2-4.6).

These are added one at a time following the build order in §10:

  * ``apply_gated_update``   - confidence-gated weight update (§4.6)  [step 1/3]
  * ``update_firing_rates``  - firing-rate EMA (§4.2)                 [step 1]
  * ``update_eligibility``   - eligibility trace (§4.3)               [step 2]
  * ``update_confidence``    - three-factor confidence rule (§4.5)    [step 3]
"""

from __future__ import annotations


def update_firing_rates(net, beta):
    """Firing-rate EMA, every step, all neurons (§4.2).

        r_j <- (1 - beta) * r_j + beta * a_j
    """
    for n in net.neurons:
        n.firing_rate = (1.0 - beta) * n.firing_rate + beta * n.activation


def update_eligibility(net, lambda_e):
    """Eligibility trace, every step, all live synapses (§4.3).

        coact = a_pre * a_post                       # rate-space "fired together"
        e_ij  <- lambda_e * e_ij + (1 - lambda_e) * coact

    Input neurons hold raw (possibly negative) coordinates, so ``coact`` can be
    negative; we clamp to keep the stated invariant ``e >= 0`` ("fired together
    recently" is a non-negative quantity).
    """
    for syn in net.synapses.values():
        coact = net.neurons[syn.pre].activation * net.neurons[syn.post].activation
        e = lambda_e * syn.eligibility + (1.0 - lambda_e) * coact
        syn.eligibility = e if e > 0.0 else 0.0


def update_confidence(net, loss, gamma_dec, gamma_q, gamma_h, e_half=0.1):
    """Three-factor confidence rule, every step, all live synapses (§4.5).

        g     = exp(-L)                                 # global outcome quality in (0, 1]
        gate  = e_ij / (e_ij + e_half)                  # bounded co-activation gate in [0,1)
        c_ij <- (1 - gamma_dec) * c_ij + gate * (gamma_q * g + gamma_h)
        c_ij <- max(c_ij, 0)

      * the decay term lets confidence fall when not reinforced;
      * ``gate`` gates the credit - only co-active synapses gain confidence;
      * ``gamma_q * g`` is the low-error driver, ``gamma_h`` the Hebbian baseline.

    DEVIATION FROM SPEC (§4.5): the spec multiplies the credit by the raw
    eligibility ``e_ij``. Empirically that is fatal on non-trivial tasks: with
    ReLU activations eligibility runs hot and unbounded, so the Hebbian term
    drives confidence into the hundreds *before* the network has learned the
    task, freezing half-trained synapses that slow decay can never release. We
    instead read eligibility as the *gate* the spec's own prose calls it -
    bounded in [0,1) via a saturating function - so per-step credit is capped at
    the drive and consolidation lags learning instead of pre-empting it.
    """
    import math
    g = math.exp(-loss)
    drive = gamma_q * g + gamma_h
    for syn in net.synapses.values():
        gate = syn.eligibility / (syn.eligibility + e_half)
        c = (1.0 - gamma_dec) * syn.confidence + gate * drive
        syn.confidence = c if c > 0.0 else 0.0


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
