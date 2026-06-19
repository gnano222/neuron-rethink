import numpy as np

from sprout.network import Network, build_graph, init_weights
from sprout.learning import apply_gated_update, update_firing_rates


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


def _meter_net(weight, confidence, grad_signed, grad_mag):
    """A one-synapse net with the gradient meters (S, M) pre-set on wire 0->1."""
    net = _one_synapse_net(confidence=confidence, weight=weight)
    s = net.synapses[(0, 1)]
    s.grad_signed = grad_signed
    s.grad_mag = grad_mag
    return net


def test_currency_optimizer_steps_along_S_over_M():
    # currency mode steps along the METERS (S/M), ignoring the instantaneous g
    net = _meter_net(weight=1.0, confidence=0.0, grad_signed=0.4, grad_mag=0.5)
    grad_w = {(0, 1): 99.0, (1, 2): 0.0, (1, 3): 0.0}   # huge g; must be ignored
    grad_b = {1: 0.0, 2: 0.0, 3: 0.0}
    apply_gated_update(net, grad_w, grad_b, eta_base=0.1, optimizer="currency")
    # step = eta/(1+c) * S/(M+eps) = 0.1 * (0.4/0.5) = 0.08 ; w <- 1.0 - 0.08
    assert abs(net.synapses[(0, 1)].weight - 0.92) < 1e-9


def test_currency_optimizer_is_magnitude_normalized():
    # perfectly consistent feedback (|S| = M) at very different scales -> identical
    # step ~ eta (decoupled from gradient magnitude: the adaptive-LR effect)
    small = _meter_net(1.0, 0.0, grad_signed=0.001, grad_mag=0.001)
    big = _meter_net(1.0, 0.0, grad_signed=10.0, grad_mag=10.0)
    gw = {(0, 1): 0.0, (1, 2): 0.0, (1, 3): 0.0}
    gb = {1: 0.0, 2: 0.0, 3: 0.0}
    apply_gated_update(small, gw, gb, eta_base=0.1, optimizer="currency")
    apply_gated_update(big, gw, gb, eta_base=0.1, optimizer="currency")
    ms = abs(1.0 - small.synapses[(0, 1)].weight)
    mb = abs(1.0 - big.synapses[(0, 1)].weight)
    assert abs(ms - mb) < 1e-9 and abs(ms - 0.1) < 1e-3


def test_currency_optimizer_holds_inconsistent_wire():
    # oscillating gradient: |S| << M -> kappa ~ 0 -> step ~ 0 (auto-anneal)
    net = _meter_net(1.0, 0.0, grad_signed=0.0, grad_mag=1.0)
    gw = {(0, 1): 5.0, (1, 2): 0.0, (1, 3): 0.0}
    gb = {1: 0.0, 2: 0.0, 3: 0.0}
    apply_gated_update(net, gw, gb, eta_base=0.1, optimizer="currency")
    assert abs(net.synapses[(0, 1)].weight - 1.0) < 1e-9


def test_currency_optimizer_defaults_to_sgd():
    # default optimizer is sgd -> instantaneous-gradient step, meters ignored
    net = _meter_net(1.0, 0.0, grad_signed=0.4, grad_mag=0.5)
    gw = {(0, 1): 2.0, (1, 2): 0.0, (1, 3): 0.0}
    gb = {1: 0.0, 2: 0.0, 3: 0.0}
    apply_gated_update(net, gw, gb, eta_base=0.1)        # no optimizer arg
    assert abs(net.synapses[(0, 1)].weight - 0.8) < 1e-12   # 1 - 0.1*2.0


def test_currency_optimizer_bias_still_ungated_plain_sgd():
    net = _meter_net(1.0, 9.0, grad_signed=0.0, grad_mag=1.0)
    net.neurons[1].bias = 1.0
    gw = {(0, 1): 0.0, (1, 2): 0.0, (1, 3): 0.0}
    gb = {1: 2.0, 2: 0.0, 3: 0.0}
    apply_gated_update(net, gw, gb, eta_base=0.1, optimizer="currency")
    assert abs(net.neurons[1].bias - 0.8) < 1e-12       # bias = plain SGD, ungated


def test_unknown_optimizer_raises():
    import pytest
    net = _meter_net(1.0, 0.0, 0.1, 0.1)
    with pytest.raises(ValueError):
        apply_gated_update(net, {(0, 1): 0.0, (1, 2): 0.0, (1, 3): 0.0},
                           {1: 0.0, 2: 0.0, 3: 0.0}, eta_base=0.1, optimizer="nope")


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
