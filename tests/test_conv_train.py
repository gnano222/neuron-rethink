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


# -- Stage D: phasic rewire (head + conv filter birth/death) ------------------

def test_phasic_rewire_fires_and_changes_structure():
    X, y = _synth(120, 0)
    from sprout.conv import ConvEconomy
    from sprout.network import build_graph, init_weights
    conv = ConvEconomy(k_max=8, kh=3, kw=3, k_init=4, seed=0)   # room to grow
    head = build_graph([conv.feat_dim(8, 8), 12, 2], density=0.5, seed=0)
    init_weights(head, seed=0)
    model = ConvModel(conv, head, 8, 8)
    cfg = Config(eta_base=0.05, enable_confidence=True, enable_prune=True,
                 enable_grow=True, phasic_structure=True, startle=False,
                 sleep_warmup=200, sleep_patience=100)
    tr = ConvTrainer(cfg, model, X, y, seed=0, conv_structure=True,
                     conv_k_max=8, conv_k_min=2)
    for _ in range(3000):
        tr.step()
    types = {e["type"] for e in tr.events}
    assert "sleep" in types                       # a plateau rewire fired
    assert ("conv_grow" in types) or ("conv_prune" in types)
    assert 2 <= conv.n_active <= 8


def test_phasic_rewire_deterministic():
    def run():
        X, y = _synth(80, 1)
        from sprout.conv import ConvEconomy
        from sprout.network import build_graph, init_weights
        conv = ConvEconomy(k_max=6, kh=3, kw=3, k_init=3, seed=3)
        head = build_graph([conv.feat_dim(8, 8), 8, 2], density=0.5, seed=3)
        init_weights(head, seed=3)
        model = ConvModel(conv, head, 8, 8)
        cfg = Config(eta_base=0.05, enable_confidence=True, enable_prune=True,
                     enable_grow=True, phasic_structure=True, startle=False,
                     sleep_warmup=150, sleep_patience=80)
        tr = ConvTrainer(cfg, model, X, y, seed=3, conv_structure=True)
        for _ in range(1500):
            tr.step()
        return conv.n_active, len(head.synapses), conv.theta.copy()

    a = run()
    b = run()
    assert a[0] == b[0] and a[1] == b[1] and np.allclose(a[2], b[2])


# -- filter-LR consolidation schedule ----------------------------------------

def test_conv_eta_schedules():
    model = _model(4, 8, 8, 8, 2, seed=0)
    cfg = Config(eta_base=0.05, enable_confidence=True)
    # cosine: full at start, ~half at midpoint, ~0 at the end
    tr = ConvTrainer(cfg, model, _synth(20, 0)[0], _synth(20, 0)[1], seed=0,
                     conv_eta=0.02, conv_eta_schedule="cosine", total_steps=1000)
    tr.step_idx = 0;    assert abs(tr._conv_eta_now() - 0.02) < 1e-9
    tr.step_idx = 500;  assert abs(tr._conv_eta_now() - 0.01) < 1e-3
    tr.step_idx = 1000; assert tr._conv_eta_now() < 1e-4
    # freeze: full until the fraction, then exactly zero
    tr2 = ConvTrainer(cfg, model, _synth(20, 0)[0], _synth(20, 0)[1], seed=0,
                      conv_eta=0.02, conv_eta_schedule="freeze", total_steps=1000,
                      freeze_frac=0.6)
    tr2.step_idx = 500; assert tr2._conv_eta_now() == 0.02
    tr2.step_idx = 700; assert tr2._conv_eta_now() == 0.0
    # none: constant
    tr3 = ConvTrainer(cfg, model, _synth(20, 0)[0], _synth(20, 0)[1], seed=0,
                      conv_eta=0.02, total_steps=1000)
    tr3.step_idx = 999; assert tr3._conv_eta_now() == 0.02
