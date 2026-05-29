import numpy as np
import pytest

from sprout.network import Network, build_graph, init_weights


# ----------------------------------------------------------------------------
# Graph construction (§5.1)
# ----------------------------------------------------------------------------

def test_build_graph_neuron_counts_and_layers():
    net = build_graph([2, 8, 8, 6, 2], density=0.5, seed=0)
    assert [len(layer) for layer in net.layers] == [2, 8, 8, 6, 2]
    assert net.num_neurons == 2 + 8 + 8 + 6 + 2


def test_every_synapse_goes_to_a_later_layer():
    # The DAG invariant: signal only ever flows forward.
    net = build_graph([2, 8, 8, 6, 2], density=0.5, seed=1)
    for (pre, post), syn in net.synapses.items():
        assert net.neurons[pre].layer < net.neurons[post].layer


def test_initial_edges_are_between_consecutive_layers():
    net = build_graph([2, 8, 8, 6, 2], density=0.5, seed=2)
    for (pre, post) in net.synapses:
        assert net.neurons[post].layer - net.neurons[pre].layer == 1


def test_min_fan_in_at_least_two_for_non_input():
    net = build_graph([2, 8, 8, 6, 2], density=0.3, seed=3)
    for nid, neuron in enumerate(net.neurons):
        if neuron.layer > 0:
            assert len(net.incoming[nid]) >= 2, f"neuron {nid} starved of input"


def test_min_fan_out_at_least_one_for_non_output():
    net = build_graph([2, 8, 8, 6, 2], density=0.3, seed=4)
    last_layer = len(net.layers) - 1
    for nid, neuron in enumerate(net.neurons):
        if neuron.layer < last_layer:
            assert len(net.outgoing[nid]) >= 1, f"neuron {nid} has no output"


def test_density_is_roughly_respected():
    net = build_graph([2, 40, 40, 2], density=0.5, seed=5)
    # count edges into the big middle layer; possible = 40*40
    edges = sum(1 for (_, post) in net.synapses if net.neurons[post].layer == 2)
    frac = edges / (40 * 40)
    assert 0.4 < frac < 0.65


def test_build_graph_deterministic():
    a = build_graph([2, 8, 6, 2], density=0.5, seed=9)
    b = build_graph([2, 8, 6, 2], density=0.5, seed=9)
    assert set(a.synapses.keys()) == set(b.synapses.keys())


# ----------------------------------------------------------------------------
# Weight init (§5.2): He using the *real* sparse fan-in
# ----------------------------------------------------------------------------

def test_init_weights_he_scaling_uses_real_fan_in():
    # Dense wide layer => k = 50 inputs per neuron; He std = sqrt(2/50).
    net = build_graph([50, 40, 2], density=1.0, seed=0)
    init_weights(net, seed=0)
    ws = [syn.weight for (_, post), syn in net.synapses.items()
          if net.neurons[post].layer == 1]
    target = np.sqrt(2.0 / 50.0)
    assert abs(np.std(ws) - target) < 0.03


def test_init_weights_scale_grows_as_fan_in_shrinks():
    # Sparser graph -> smaller k -> larger per-weight std (variance ~ 2/k).
    dense = build_graph([50, 40, 2], density=1.0, seed=1)
    sparse = build_graph([50, 40, 2], density=0.4, seed=1)
    init_weights(dense, seed=1)
    init_weights(sparse, seed=1)
    std_dense = np.std([s.weight for s in dense.synapses.values()])
    std_sparse = np.std([s.weight for s in sparse.synapses.values()])
    assert std_sparse > std_dense


# ----------------------------------------------------------------------------
# Forward pass (§4.1): ReLU hidden + softmax output
# ----------------------------------------------------------------------------

def _hand_net():
    """A fully-specified 2-2-2 net so we can compute the forward pass by hand."""
    net = Network([2, 2, 2])
    # input ids 0,1 ; hidden ids 2,3 ; output ids 4,5
    net.add_synapse(0, 2, weight=1.0)
    net.add_synapse(1, 2, weight=-1.0)
    net.add_synapse(0, 3, weight=0.5)
    net.add_synapse(1, 3, weight=0.5)
    net.add_synapse(2, 4, weight=2.0)
    net.add_synapse(3, 4, weight=0.0)
    net.add_synapse(2, 5, weight=0.0)
    net.add_synapse(3, 5, weight=1.0)
    return net


def test_forward_relu_then_softmax_hand_computed():
    net = _hand_net()
    probs = net.forward(np.array([1.0, 0.0]))
    # hidden z2 = 1*1 + 0*-1 = 1 -> relu 1 ; z3 = 1*.5 + 0*.5 = .5 -> relu .5
    # out z4 = 2*1 = 2 ; z5 = 1*.5 = .5 ; softmax([2, .5])
    z = np.array([2.0, 0.5])
    expected = np.exp(z) / np.exp(z).sum()
    assert np.allclose(probs, expected)


def test_forward_relu_clamps_negatives():
    net = _hand_net()
    net.forward(np.array([0.0, 1.0]))
    # z2 = -1 -> relu 0 ; the hidden neuron with id 2 must read exactly 0
    assert net.neurons[2].activation == 0.0


def test_forward_probs_sum_to_one():
    net = build_graph([2, 8, 6, 2], density=0.6, seed=0)
    init_weights(net, seed=0)
    probs = net.forward(np.array([0.3, -0.7]))
    assert probs.shape == (2,)
    assert abs(probs.sum() - 1.0) < 1e-9


# ----------------------------------------------------------------------------
# Backward pass (§4.4): finite-difference gradient check (the gold standard)
# ----------------------------------------------------------------------------

def test_backward_matches_finite_differences():
    net = build_graph([2, 4, 3, 2], density=1.0, seed=0)
    init_weights(net, seed=0)
    # give biases something non-trivial too
    rng = np.random.default_rng(0)
    for n in net.neurons:
        if n.layer > 0:
            n.bias = float(rng.normal(0, 0.3))

    x = np.array([0.6, -0.4])
    y_true = 1

    net.forward(x)
    loss, grad_w, grad_b = net.backward(y_true)

    eps = 1e-6

    def loss_at():
        p = net.forward(x)
        return -np.log(p[y_true] + 1e-12)

    # check weight grads
    for key, syn in net.synapses.items():
        w0 = syn.weight
        syn.weight = w0 + eps
        lp = loss_at()
        syn.weight = w0 - eps
        lm = loss_at()
        syn.weight = w0
        fd = (lp - lm) / (2 * eps)
        assert abs(fd - grad_w[key]) < 1e-4, f"weight grad mismatch at {key}"

    # check bias grads
    for nid, neuron in enumerate(net.neurons):
        if neuron.layer == 0:
            continue
        b0 = neuron.bias
        neuron.bias = b0 + eps
        lp = loss_at()
        neuron.bias = b0 - eps
        lm = loss_at()
        neuron.bias = b0
        fd = (lp - lm) / (2 * eps)
        assert abs(fd - grad_b[nid]) < 1e-4, f"bias grad mismatch at neuron {nid}"


def test_backward_reports_cross_entropy_loss():
    net = _hand_net()
    p = net.forward(np.array([1.0, 0.0]))
    loss, _, _ = net.backward(0)
    assert abs(loss - (-np.log(p[0]))) < 1e-9


def test_gradients_only_exist_for_live_synapses():
    net = build_graph([2, 6, 4, 2], density=0.5, seed=0)
    init_weights(net, seed=0)
    net.forward(np.array([0.2, 0.9]))
    _, grad_w, _ = net.backward(0)
    assert set(grad_w.keys()) == set(net.synapses.keys())
