import numpy as np

from sprout.network import Network, build_graph, init_weights
from sprout.learning import (
    apply_gated_update,
    update_firing_rates,
    update_eligibility,
    update_confidence,
)


def _one_synapse_net(confidence=0.0, weight=0.0):
    net = Network([1, 1, 2])  # ids: input 0, hidden 1, outputs 2,3
    # we only care about synapse 0->1 for the gating test
    net.add_synapse(0, 1, weight=weight, confidence=confidence)
    net.add_synapse(1, 2, weight=0.0)
    net.add_synapse(1, 3, weight=0.0)
    return net


def test_gated_update_plain_sgd_when_confidence_zero():
    net = _one_synapse_net(confidence=0.0, weight=1.0)
    grad_w = {(0, 1): 2.0, (1, 2): 0.0, (1, 3): 0.0}
    grad_b = {1: 0.0, 2: 0.0, 3: 0.0}
    apply_gated_update(net, grad_w, grad_b, eta_base=0.1)
    # w <- 1.0 - 0.1 * 2.0 = 0.8  (no gating)
    assert abs(net.synapses[(0, 1)].weight - 0.8) < 1e-12


def test_gated_update_confident_synapse_barely_moves():
    net = _one_synapse_net(confidence=9.0, weight=1.0)
    grad_w = {(0, 1): 2.0, (1, 2): 0.0, (1, 3): 0.0}
    grad_b = {1: 0.0, 2: 0.0, 3: 0.0}
    apply_gated_update(net, grad_w, grad_b, eta_base=0.1)
    # eta_eff = 0.1 / (1 + 9) = 0.01 ; w <- 1.0 - 0.01*2.0 = 0.98
    assert abs(net.synapses[(0, 1)].weight - 0.98) < 1e-12


def test_more_confidence_means_smaller_step():
    lo = _one_synapse_net(confidence=1.0, weight=1.0)
    hi = _one_synapse_net(confidence=5.0, weight=1.0)
    gw = {(0, 1): 1.0, (1, 2): 0.0, (1, 3): 0.0}
    gb = {1: 0.0, 2: 0.0, 3: 0.0}
    apply_gated_update(lo, gw, gb, eta_base=0.1)
    apply_gated_update(hi, gw, gb, eta_base=0.1)
    move_lo = abs(1.0 - lo.synapses[(0, 1)].weight)
    move_hi = abs(1.0 - hi.synapses[(0, 1)].weight)
    assert move_hi < move_lo


def test_biases_use_base_rate_ungated():
    net = _one_synapse_net(confidence=9.0, weight=1.0)
    net.neurons[1].bias = 1.0
    grad_w = {(0, 1): 0.0, (1, 2): 0.0, (1, 3): 0.0}
    grad_b = {1: 2.0, 2: 0.0, 3: 0.0}
    apply_gated_update(net, grad_w, grad_b, eta_base=0.1)
    # bias is ungated even though the neuron's incoming synapse is confident
    assert abs(net.neurons[1].bias - (1.0 - 0.1 * 2.0)) < 1e-12


def test_firing_rate_ema_update():
    net = Network([2, 2, 2])
    net.neurons[2].firing_rate = 0.0
    net.neurons[2].activation = 1.0
    update_firing_rates(net, beta=0.05)
    assert abs(net.neurons[2].firing_rate - 0.05) < 1e-12


def test_firing_rate_ema_converges_to_constant_activation():
    net = Network([1, 1, 2])
    net.neurons[1].firing_rate = 0.0
    net.neurons[1].activation = 0.3
    for _ in range(500):
        update_firing_rates(net, beta=0.05)
    assert abs(net.neurons[1].firing_rate - 0.3) < 1e-3


def test_eligibility_ema_update():
    net = Network([1, 1, 2])
    syn = net.add_synapse(0, 1, weight=0.0)
    net.neurons[0].activation = 1.0  # pre
    net.neurons[1].activation = 0.5  # post
    update_eligibility(net, lambda_e=0.9)
    # e <- 0.9*0 + 0.1*(1.0*0.5) = 0.05
    assert abs(syn.eligibility - 0.05) < 1e-12


def test_eligibility_high_on_coactive_low_on_inactive():
    net = Network([2, 2, 2])
    coactive = net.add_synapse(0, 2)   # both endpoints will fire
    silent = net.add_synapse(1, 3)     # post stays silent
    net.neurons[0].activation = 1.0
    net.neurons[2].activation = 1.0
    net.neurons[1].activation = 1.0
    net.neurons[3].activation = 0.0    # inactive post
    for _ in range(50):
        update_eligibility(net, lambda_e=0.9)
    assert coactive.eligibility > 0.5
    assert silent.eligibility < 1e-6


def test_eligibility_decays_when_coactivation_stops():
    net = Network([1, 1, 2])
    syn = net.add_synapse(0, 1)
    net.neurons[0].activation = 1.0
    net.neurons[1].activation = 1.0
    for _ in range(50):
        update_eligibility(net, lambda_e=0.9)
    charged = syn.eligibility
    net.neurons[1].activation = 0.0  # stop co-firing
    for _ in range(50):
        update_eligibility(net, lambda_e=0.9)
    assert syn.eligibility < 0.1 * charged


def test_eligibility_never_negative():
    # input coords can be negative => coact can be negative; invariant e>=0 holds.
    net = Network([1, 1, 2])
    syn = net.add_synapse(0, 1)
    net.neurons[0].activation = -2.0  # negative input "activation"
    net.neurons[1].activation = 1.0
    for _ in range(20):
        update_eligibility(net, lambda_e=0.9)
    assert syn.eligibility >= 0.0


# The eligibility enters as a *bounded gate*  gate = e / (e + e_half)  in [0,1),
# not as an unbounded multiplier. Without the bound, hot eligibility drives
# confidence into the hundreds and freezes synapses before the task is learned.
def test_confidence_rises_for_coactive_low_loss():
    net = Network([1, 1, 2])
    syn = net.add_synapse(0, 1)
    syn.eligibility = 0.5
    update_confidence(net, loss=0.0, gamma_dec=0.01, gamma_q=0.05, gamma_h=0.01, e_half=0.1)
    # gate = 0.5/0.6 = 5/6 ; g=1 ; c <- 0.99*0 + (5/6)*(0.05+0.01) = 0.05
    assert abs(syn.confidence - 0.05) < 1e-9


def test_eligibility_gates_confidence_credit():
    # A synapse that never co-fired (e=0) gains no confidence even at low loss.
    net = Network([2, 2, 2])
    coactive = net.add_synapse(0, 2)
    silent = net.add_synapse(1, 3)
    coactive.eligibility = 0.5
    silent.eligibility = 0.0
    for _ in range(200):
        update_confidence(net, loss=0.0, gamma_dec=0.01, gamma_q=0.05, gamma_h=0.01, e_half=0.1)
    assert coactive.confidence > 0.5
    assert silent.confidence == 0.0


def test_confidence_credit_is_bounded_by_gate():
    # Huge eligibility cannot add more than ~the drive per step (the gate -> 1).
    # This is what prevents premature freezing on hard tasks.
    big = Network([1, 1, 2]); sbig = big.add_synapse(0, 1); sbig.eligibility = 1000.0
    mod = Network([1, 1, 2]); smod = mod.add_synapse(0, 1); smod.eligibility = 10.0
    update_confidence(big, loss=0.0, gamma_dec=0.01, gamma_q=0.05, gamma_h=0.01, e_half=0.1)
    update_confidence(mod, loss=0.0, gamma_dec=0.01, gamma_q=0.05, gamma_h=0.01, e_half=0.1)
    drive = 0.06
    assert sbig.confidence < drive + 1e-9          # capped by the drive
    assert abs(sbig.confidence - smod.confidence) < 0.01  # 100x elig => ~same credit


def test_low_loss_drives_more_confidence_than_high_loss():
    good = Network([1, 1, 2]); sg = good.add_synapse(0, 1); sg.eligibility = 1.0
    bad = Network([1, 1, 2]); sb = bad.add_synapse(0, 1); sb.eligibility = 1.0
    update_confidence(good, loss=0.0, gamma_dec=0.01, gamma_q=0.05, gamma_h=0.01, e_half=0.1)
    update_confidence(bad, loss=5.0, gamma_dec=0.01, gamma_q=0.05, gamma_h=0.01, e_half=0.1)
    assert sg.confidence > sb.confidence


def test_confidence_decays_when_not_reinforced():
    net = Network([1, 1, 2])
    syn = net.add_synapse(0, 1)
    syn.confidence = 2.0
    syn.eligibility = 0.0  # no longer co-active -> gate 0 -> pure decay
    update_confidence(net, loss=0.0, gamma_dec=0.01, gamma_q=0.05, gamma_h=0.01, e_half=0.1)
    # pure decay: c <- 0.99 * 2.0 = 1.98
    assert abs(syn.confidence - 1.98) < 1e-9
    assert syn.confidence < 2.0


def test_confidence_never_negative():
    net = Network([1, 1, 2])
    syn = net.add_synapse(0, 1)
    syn.confidence = 0.0
    syn.eligibility = 0.0
    for _ in range(50):
        update_confidence(net, loss=10.0, gamma_dec=0.5, gamma_q=0.05, gamma_h=0.01, e_half=0.1)
    assert syn.confidence >= 0.0


def test_training_overfits_a_single_point():
    # Whole pipeline: forward -> backward -> update should drive loss down on
    # one fixed sample (validates the loop end to end).
    net = build_graph([2, 6, 4, 2], density=1.0, seed=0)
    init_weights(net, seed=0)
    x, y = np.array([0.5, -0.3]), 1

    net.forward(x)
    loss0, _, _ = net.backward(y)
    for _ in range(200):
        net.forward(x)
        _, gw, gb = net.backward(y)
        apply_gated_update(net, gw, gb, eta_base=0.1)
    net.forward(x)
    loss1, _, _ = net.backward(y)

    assert loss1 < 0.1 * loss0
