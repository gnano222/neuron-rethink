import numpy as np

from sprout.network import build_graph, init_weights
from sprout.fast import ArrayNet, _settledness_vec, train_array
from sprout import learning, currency
from sprout.currency import settledness
from sprout.train import Config, accuracy
from sprout.data import generate_spirals


def _net(seed=0, layers=(4, 6, 3), density=0.6):
    net = build_graph(list(layers), density=density, seed=seed)
    init_weights(net, seed=seed)
    return net


def _cfg():
    return Config(eta_base=0.02, enable_confidence=True, enable_prune=True,
                  enable_grow=True, gamma_dec=0.001, t_struct=200,
                  phasic_structure=True, startle=True, grow_demand_k=4)


def _obj_wake_step(net, cfg, x, y, step_idx):
    """Object-side wake step mirroring Trainer._step_currency order (no structure)."""
    net.forward(x)
    learning.update_firing_rates(net, cfg.beta)
    loss, gw, gb = net.backward(int(y))
    currency.update_gradient_meters(net, gw, cfg.beta_g, step_idx=step_idx,
                                    lazy=cfg.lazy_meters)
    if cfg.enable_confidence and cfg.confidence_mode == "twod":
        currency.update_confidence_2d(net, cfg.conf_gain, cfg.conf_alpha, cfg.c_max,
                                      cfg.settled_mode, cfg.conf_k)
    learning.apply_gated_update(net, gw, gb, cfg.eta_base)
    for syn in net.synapses.values():
        syn.age += 1
    return loss


# -- Task 1: representation + round-trip --------------------------------------

def test_from_network_and_sync_roundtrip():
    net = _net()
    an = ArrayNet.from_network(net)
    assert an.keys == list(net.synapses.keys())
    assert an.E == len(net.synapses) and an.N == net.num_neurons
    an.weight += 1.5
    an.confidence[:] = 0.25
    an.age += 3
    an.sync_into(net)
    for r, k in enumerate(an.keys):
        s = net.synapses[k]
        assert abs(s.weight - an.weight[r]) < 1e-12
        assert abs(s.confidence - 0.25) < 1e-12 and s.age == an.age[r]


# -- Task 2: forward parity ---------------------------------------------------

def test_forward_matches_object():
    net = _net(layers=(4, 8, 5, 3), density=0.5)
    an = ArrayNet.from_network(net)
    rng = np.random.default_rng(1)
    for _ in range(5):
        x = rng.normal(size=4)
        p = net.forward(x)
        pa = an.forward(x)
        assert np.allclose(p, pa, atol=1e-9)
        obj_a = np.array([n.activation for n in net.neurons])
        assert np.allclose(obj_a, an.activation, atol=1e-9)


# -- Task 3: backward parity --------------------------------------------------

def test_backward_matches_object():
    net = _net(layers=(4, 8, 5, 3), density=0.5)
    an = ArrayNet.from_network(net)
    x = np.random.default_rng(2).normal(size=4)
    y = 1
    net.forward(x)
    loss_o, gw_o, gb_o = net.backward(y)
    an.forward(x)
    loss_a, gw_a = an.backward(y)
    assert abs(loss_o - loss_a) < 1e-9
    gw_o_arr = np.array([gw_o[k] for k in an.keys])
    assert np.allclose(gw_o_arr, gw_a, atol=1e-9)
    for nid, g in gb_o.items():
        assert abs(g - an.delta[nid]) < 1e-9


# -- Task 4: settledness + multi-step wake parity -----------------------------

def test_settledness_vec_matches_scalar():
    d = np.linspace(0.0, 3.0, 41)
    for mode in ("hard", "sigmoid", "exp", "rational"):
        vec = _settledness_vec(d, mode, 3.0)
        scal = np.array([settledness(di, mode, 3.0) for di in d])
        assert np.allclose(vec, scal, atol=1e-12)


def test_wake_step_parity():
    cfg = _cfg()
    net = _net(layers=(4, 8, 6, 3), density=0.5)
    an = ArrayNet.from_network(net)
    rng = np.random.default_rng(7)
    X = rng.normal(size=(64, 4))
    Y = rng.integers(0, 3, 64)
    stream = rng.integers(0, 64, size=150)
    for t, s in enumerate(stream):
        _obj_wake_step(net, cfg, X[s], int(Y[s]), t)
        an.step(X[s], int(Y[s]), cfg, t)
    obj_w = np.array([net.synapses[k].weight for k in an.keys])
    obj_c = np.array([net.synapses[k].confidence for k in an.keys])
    obj_m = np.array([net.synapses[k].grad_mag for k in an.keys])
    assert np.allclose(obj_w, an.weight, atol=1e-6)
    assert np.allclose(obj_c, an.confidence, atol=1e-6)
    assert np.allclose(obj_m, an.grad_mag, atol=1e-6)
    # predictions identical on a probe set (the real bar)
    for x in X[:20]:
        assert net.forward(x).argmax() == an.forward(x).argmax()


# -- Task 5: end-to-end train_array -------------------------------------------

def test_train_array_learns_and_sparsifies():
    cfg = _cfg()
    cfg.sleep_warmup = 500
    cfg.sleep_patience = 300
    X, y = generate_spirals(n=400, seed=0)
    net = build_graph([2, 16, 16, 16, 2], density=0.5, seed=0)
    init_weights(net, seed=0)
    start = len(net.synapses)
    net = train_array(cfg, net, X, y, seed=0, steps=15000)
    assert accuracy(net, X, y) > 0.7
    assert len(net.synapses) < start


def test_train_array_deterministic():
    cfg = _cfg()
    cfg.sleep_warmup = 500
    cfg.sleep_patience = 300
    X, y = generate_spirals(n=300, seed=1)

    def run():
        net = build_graph([2, 10, 2], density=0.5, seed=3)
        init_weights(net, seed=3)
        return train_array(cfg, net, X, y, seed=3, steps=2000)

    a, b = run(), run()
    ka = list(a.synapses.keys())
    assert ka == list(b.synapses.keys())
    assert np.allclose([a.synapses[k].weight for k in ka],
                       [b.synapses[k].weight for k in ka], atol=1e-12)
