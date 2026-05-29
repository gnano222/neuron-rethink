import numpy as np

from sprout.network import Network, build_graph, init_weights
from sprout.learning import update_firing_rates
from sprout.plasticity import homeostasis, prune, grow


def test_homeostasis_scales_down_overactive_neuron():
    net = Network([1, 1, 2])
    syn = net.add_synapse(0, 1, weight=1.0)
    net.neurons[1].firing_rate = 1.5  # well above r_target
    homeostasis(net, r_target=0.15, rho=0.1, eps=1e-6)
    assert syn.weight < 1.0


def test_homeostasis_scales_up_underactive_neuron():
    net = Network([1, 1, 2])
    syn = net.add_synapse(0, 1, weight=1.0)
    net.neurons[1].firing_rate = 0.01  # below r_target
    homeostasis(net, r_target=0.15, rho=0.1, eps=1e-6)
    assert syn.weight > 1.0


def test_homeostasis_is_gentle():
    # 10x over target should NOT slam the weight to 1/10; rho makes it gentle.
    net = Network([1, 1, 2])
    syn = net.add_synapse(0, 1, weight=1.0)
    net.neurons[1].firing_rate = 1.5  # 10x r_target
    homeostasis(net, r_target=0.15, rho=0.1, eps=1e-6)
    # scale = (0.15/1.5)^0.1 ~= 0.794
    assert 0.7 < syn.weight < 0.85


def test_homeostasis_no_change_at_target():
    net = Network([1, 1, 2])
    syn = net.add_synapse(0, 1, weight=1.0)
    net.neurons[1].firing_rate = 0.15
    homeostasis(net, r_target=0.15, rho=0.1, eps=1e-6)
    assert abs(syn.weight - 1.0) < 1e-6


def test_homeostasis_clamps_runaway_scale_on_dead_neuron():
    # A dead ReLU neuron (r~0) would otherwise get scale=(0.15/1e-6)^0.1~=3.3,
    # which compounds into divergence. The clamp must cap the per-round scale.
    net = Network([1, 1, 2])
    syn = net.add_synapse(0, 1, weight=1.0)
    net.neurons[1].firing_rate = 0.0
    homeostasis(net, r_target=0.15, rho=0.1, eps=1e-6, scale_max=1.5)
    assert syn.weight <= 1.5 + 1e-9


def test_homeostasis_drives_firing_rate_toward_target():
    # Closed loop: forward -> firing-rate EMA -> homeostasis should pull an
    # over-active neuron's firing rate down toward r_target.
    net = Network([1, 1, 2])
    net.add_synapse(0, 1, weight=2.0)
    net.neurons[1].firing_rate = 2.0
    for _ in range(400):
        net.forward(np.array([1.0]))      # input = 1 ; hidden a = w*1
        update_firing_rates(net, beta=0.3)
        homeostasis(net, r_target=0.15, rho=0.3, eps=1e-6)
    assert 0.10 < net.neurons[1].firing_rate < 0.20


# ----------------------------------------------------------------------------
# Pruning by utility (§4.7)
# ----------------------------------------------------------------------------

def _two_input_net(w_low=0.001, w_high=2.0, age_low=300, age_high=300,
                   r_pre=0.1):
    """Post neuron (id 2) fed by a low-utility (0->2) and high-utility (1->2)
    synapse, so the low one can be pruned without orphaning the post."""
    net = Network([2, 1, 2])
    low = net.add_synapse(0, 2, weight=w_low, age=age_low)
    high = net.add_synapse(1, 2, weight=w_high, age=age_high)
    net.neurons[0].firing_rate = r_pre
    net.neurons[1].firing_rate = r_pre
    return net, low, high


def test_prune_removes_low_utility_past_grace():
    net, low, high = _two_input_net(w_low=0.001, r_pre=0.1)  # u_low = 1e-4
    pruned = prune(net, theta_prune=0.01, t_grace=200, max_prune=2)
    assert (0, 2) in pruned
    assert (0, 2) not in net.synapses
    assert (1, 2) in net.synapses  # high-utility survives


def test_prune_respects_grace_period():
    net, low, high = _two_input_net(w_low=0.001, age_low=100, r_pre=0.1)
    pruned = prune(net, theta_prune=0.01, t_grace=200, max_prune=2)
    assert (0, 2) not in pruned          # too young to prune
    assert (0, 2) in net.synapses


def test_prune_keeps_high_utility_synapse():
    net, low, high = _two_input_net(w_low=0.5, r_pre=0.1)  # u_low = 0.05 > 0.01
    pruned = prune(net, theta_prune=0.01, t_grace=200, max_prune=2)
    assert pruned == []
    assert (0, 2) in net.synapses


def test_prune_caps_per_round():
    # five prunable low-utility synapses; cap of 2 must hold.
    net = Network([5, 1, 2])
    for pre in range(5):
        net.add_synapse(pre, 5, weight=0.001, age=300)
        net.neurons[pre].firing_rate = 0.1
    pruned = prune(net, theta_prune=0.01, t_grace=200, max_prune=2)
    assert len(pruned) == 2
    assert len(net.incoming[5]) == 3  # 5 - 2


def test_prune_does_not_orphan_last_input():
    # A neuron's *only* incoming synapse must not be pruned (don't starve it).
    net = Network([1, 1, 2])
    net.add_synapse(0, 1, weight=0.001, age=300)
    net.neurons[0].firing_rate = 0.1
    pruned = prune(net, theta_prune=0.01, t_grace=200, max_prune=2)
    assert pruned == []
    assert (0, 1) in net.synapses


def test_prune_uses_presynaptic_firing_rate():
    # Same tiny weight, but a quiet presyn (low r) is prunable while an active
    # presyn (high r) keeps utility above threshold.
    quiet, _, _ = _two_input_net(w_low=0.05, r_pre=0.1)   # u = 0.005 < 0.01
    active, _, _ = _two_input_net(w_low=0.05, r_pre=1.0)  # u = 0.05  > 0.01
    pq = prune(quiet, theta_prune=0.01, t_grace=200, max_prune=2)
    pa = prune(active, theta_prune=0.01, t_grace=200, max_prune=2)
    assert (0, 2) in pq
    assert (0, 2) not in pa


# ----------------------------------------------------------------------------
# Growth into underfiring neurons (§4.7) - the climax
# ----------------------------------------------------------------------------

def _grow_net():
    # layer0={0,1}, layer1={2,3}, layer2={4,5}
    net = Network([2, 2, 2])
    for n in net.neurons:
        n.firing_rate = 1.0  # everyone healthy by default
    return net


def test_grow_adds_synapse_to_underfiring_neuron():
    net = _grow_net()
    net.add_synapse(0, 2)              # neuron 2 only has input from 0
    net.neurons[2].firing_rate = 0.0   # underfiring
    net.neurons[1].activation = 0.9    # the only available upstream candidate
    grown = grow(net, r_target=0.15, f_under=0.5, max_grow=2)
    assert (1, 2) in grown
    assert (1, 2) in net.synapses


def test_grow_picks_highest_activation_source():
    net = _grow_net()
    net.neurons[2].firing_rate = 0.0
    net.neurons[0].activation = 0.2
    net.neurons[1].activation = 0.95   # strongest -> should be chosen
    grown = grow(net, r_target=0.15, f_under=0.5, max_grow=1)
    assert grown == [(1, 2)]


def test_grown_synapse_is_newborn_zero_state():
    net = _grow_net()
    net.add_synapse(0, 2)
    net.neurons[2].firing_rate = 0.0
    net.neurons[1].activation = 0.9
    grow(net, r_target=0.15, f_under=0.5, max_grow=1)
    syn = net.synapses[(1, 2)]
    assert syn.weight == 0.0
    assert syn.confidence == 0.0
    assert syn.eligibility == 0.0
    assert syn.age == 0


def test_grow_respects_dag_earlier_to_later():
    net = _grow_net()
    net.neurons[4].firing_rate = 0.0   # an output neuron, underfiring
    for nid in (0, 1, 2, 3):
        net.neurons[nid].activation = 0.5
    grown = grow(net, r_target=0.15, f_under=0.5, max_grow=4)
    for (pre, post) in grown:
        assert net.neurons[pre].layer < net.neurons[post].layer


def test_grow_skips_well_firing_neuron():
    net = _grow_net()  # all firing_rate=1.0 > threshold 0.075
    net.neurons[1].activation = 0.9
    grown = grow(net, r_target=0.15, f_under=0.5, max_grow=2)
    assert grown == []


def test_grow_caps_per_round():
    net = _grow_net()
    net.neurons[2].firing_rate = 0.0
    net.neurons[3].firing_rate = 0.0
    net.neurons[4].firing_rate = 0.0
    net.neurons[5].firing_rate = 0.0
    for nid in (0, 1, 2, 3):
        net.neurons[nid].activation = 0.5
    grown = grow(net, r_target=0.15, f_under=0.5, max_grow=2)
    assert len(grown) == 2


def test_grow_skips_neuron_with_no_candidates():
    # neuron 2 already connected to every earlier-layer neuron -> nothing to add.
    net = _grow_net()
    net.add_synapse(0, 2)
    net.add_synapse(1, 2)
    net.neurons[2].firing_rate = 0.0
    grown = grow(net, r_target=0.15, f_under=0.5, max_grow=2)
    assert (2 not in [post for (_, post) in grown])


def test_grow_increments_attempt_counter():
    net = _grow_net()
    net.add_synapse(0, 2)
    net.neurons[2].firing_rate = 0.0
    net.neurons[1].activation = 0.9
    grow(net, r_target=0.15, f_under=0.5, max_grow=1)
    assert net.neurons[2].grow_attempts == 1


def test_grow_retires_neuron_after_exhausting_budget():
    # A chronically-dead neuron must stop being a growth target once it has
    # used up its budget, so growth doesn't churn forever (dead-ReLU pathology).
    net = Network([4, 1, 2])  # neuron 4 (layer1) can take up to 4 inputs
    for n in net.neurons:
        n.firing_rate = 1.0           # everyone healthy...
    net.neurons[4].firing_rate = 0.0  # ...except the dead target
    net.neurons[4].grow_attempts = 2
    for nid in range(4):
        net.neurons[nid].activation = 0.5
    grown = grow(net, r_target=0.15, f_under=0.5, max_grow=4, grow_budget=2)
    assert grown == []  # budget already spent -> retired
