import numpy as np

from sprout.network import build_graph, init_weights
from sprout.train import Config
from sprout.conv import ConvEconomy
from sprout.conv_train import ConvModel, ConvTrainer


def _synth(n, seed):
    """Vertical-bar (class 0) vs horizontal-bar (class 1) at random positions in
    an 8x8 image -- a translation-invariant pattern a conv head can learn."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 8, 8))
    y = np.zeros(n, int)
    for i in range(n):
        c = i % 2
        r, cc = int(rng.integers(0, 6)), int(rng.integers(0, 6))
        if c == 0:
            X[i, r:r + 3, cc] = 1.0           # vertical bar
        else:
            X[i, r, cc:cc + 3] = 1.0          # horizontal bar
        y[i] = c
    X += 0.01 * rng.normal(size=X.shape)
    return X, y


def _model(k_max, h, w, hidden, n_out, seed):
    conv = ConvEconomy(k_max=k_max, kh=3, kw=3, seed=seed)
    head = build_graph([conv.feat_dim(h, w), hidden, n_out], density=0.6, seed=seed)
    init_weights(head, seed=seed)
    return ConvModel(conv, head, h, w)


def test_convmodel_forward_shapes():
    model = _model(4, 10, 10, 6, 3, seed=0)
    probs, feat, cache = model.forward(np.random.default_rng(0).normal(size=(10, 10)))
    assert feat.shape == (model.conv.feat_dim(10, 10),)
    assert probs.shape == (3,)


def test_joint_grad_into_conv_matches_finite_diff():
    """The linchpin for the joint system: dL/dtheta via the head-input-delta
    bridge + conv.backward matches finite differences of the head's CE loss."""
    rng = np.random.default_rng(0)
    model = _model(2, 8, 8, 5, 3, seed=0)
    conv, head = model.conv, model.head
    img, yi = rng.normal(size=(8, 8)), 1

    probs, feat, cache = model.forward(img)
    loss, gw, gb = head.backward(yi)
    d_feat = model.input_delta(gb)
    g = conv.backward(d_feat, cache)

    def loss_fn(theta):
        conv.theta = theta
        f, _ = conv.forward(img)
        p = head.forward(f)
        return float(-np.log(p[yi] + 1e-12))

    base = conv.theta.copy()
    fd = np.zeros_like(base)
    eps = 1e-6
    it = np.nditer(base, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        pp = base.copy(); pp[idx] += eps
        mm = base.copy(); mm[idx] -= eps
        fd[idx] = (loss_fn(pp) - loss_fn(mm)) / (2 * eps)
        it.iternext()
    conv.theta = base
    assert np.allclose(g, fd, atol=1e-5)


def test_convtrainer_learns_on_synthetic():
    X, y = _synth(120, 0)
    model = _model(6, 8, 8, 12, 2, seed=0)
    cfg = Config(eta_base=0.05, enable_confidence=True,
                 enable_prune=False, enable_grow=False,
                 phasic_structure=True, startle=False)
    tr = ConvTrainer(cfg, model, X, y, seed=0)
    for _ in range(6000):
        tr.step()
    assert tr.accuracy(X, y) > 0.75            # chance = 0.5


def test_convtrainer_fixed_conv_keeps_filters_but_head_learns():
    X, y = _synth(80, 0)
    model = _model(4, 8, 8, 8, 2, seed=0)
    before_filters = model.conv.theta.copy()
    before_w = np.array([s.weight for s in model.head.synapses.values()])
    cfg = Config(eta_base=0.05, enable_confidence=True)
    tr = ConvTrainer(cfg, model, X, y, seed=0, learn_conv=False)
    for _ in range(300):
        tr.step()
    assert np.allclose(model.conv.theta, before_filters)      # filters frozen
    after_w = np.array([s.weight for s in model.head.synapses.values()])
    assert not np.allclose(after_w, before_w)                 # head still learned


def test_convtrainer_deterministic():
    def run():
        X, y = _synth(80, 1)
        model = _model(4, 8, 8, 8, 2, seed=2)
        cfg = Config(eta_base=0.05, enable_confidence=True)
        tr = ConvTrainer(cfg, model, X, y, seed=2)
        for _ in range(500):
            tr.step()
        return tr.accuracy(X, y), model.conv.theta.copy()

    a1, t1 = run()
    a2, t2 = run()
    assert a1 == a2 and np.allclose(t1, t2)
