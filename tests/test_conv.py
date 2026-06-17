import numpy as np

from sprout.conv import (conv_valid_forward, conv_valid_filter_grad,
                         maxpool_forward, maxpool_backward)


# -- Stage A: pure conv / pool math ------------------------------------------

def test_conv_forward_matches_naive_loop():
    rng = np.random.default_rng(0)
    img = rng.normal(size=(6, 7))
    kernels = rng.normal(size=(3, 3, 3))           # K=3, 3x3
    out = conv_valid_forward(img, kernels)
    oh, ow = 6 - 3 + 1, 7 - 3 + 1
    assert out.shape == (3, oh, ow)
    naive = np.zeros((3, oh, ow))
    for k in range(3):
        for i in range(oh):
            for j in range(ow):
                naive[k, i, j] = np.sum(img[i:i+3, j:j+3] * kernels[k])
    assert np.allclose(out, naive, atol=1e-12)


def test_maxpool_forward_picks_block_maxima():
    a = np.array([[[1.0, 2, 0, 0],
                   [3, 4, 0, 0],
                   [0, 0, 9, 5],
                   [0, 0, 6, 7]]])              # (1,4,4)
    pooled, argmax = maxpool_forward(a, 2)
    assert pooled.shape == (1, 2, 2) and argmax.shape == (1, 2, 2)
    assert pooled[0, 0, 0] == 4 and pooled[0, 1, 1] == 9


def test_maxpool_backward_routes_only_to_argmax():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(2, 4, 4))
    pooled, argmax = maxpool_forward(a, 2)
    d_pooled = rng.normal(size=pooled.shape)
    d_a = maxpool_backward(d_pooled, argmax, a.shape, 2)
    assert d_a.shape == a.shape
    # exactly one nonzero per 2x2 window, equal to the corresponding d_pooled
    for k in range(2):
        for i in range(2):
            for j in range(2):
                win = d_a[k, i*2:i*2+2, j*2:j*2+2]
                assert np.count_nonzero(win) == 1
                assert np.isclose(win.sum(), d_pooled[k, i, j])
    # the nonzero sits where a had its max
    for k in range(2):
        for i in range(2):
            for j in range(2):
                blk = a[k, i*2:i*2+2, j*2:j*2+2]
                pos = np.unravel_index(np.argmax(blk), blk.shape)
                assert d_a[k, i*2+pos[0], j*2+pos[1]] != 0.0


# -- the linchpin: analytic filter grad vs finite differences -----------------

def _fd_grad(loss_fn, kernels, eps=1e-6):
    g = np.zeros_like(kernels)
    it = np.nditer(kernels, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = kernels[idx]
        kernels[idx] = orig + eps
        lp = loss_fn(kernels)
        kernels[idx] = orig - eps
        lm = loss_fn(kernels)
        kernels[idx] = orig
        g[idx] = (lp - lm) / (2 * eps)
        it.iternext()
    return g


def test_filter_grad_matches_finite_diff_relu():
    rng = np.random.default_rng(2)
    img = rng.normal(size=(7, 7))
    kernels = rng.normal(size=(2, 3, 3))

    def loss_fn(kr):
        return float(np.sum(np.maximum(conv_valid_forward(img, kr), 0.0)))

    preact = conv_valid_forward(img, kernels)
    d_preact = (preact > 0).astype(float)               # d/d preact of sum(relu)
    g = conv_valid_filter_grad(img, d_preact)
    assert np.allclose(g, _fd_grad(loss_fn, kernels.copy()), atol=1e-5)


def test_filter_grad_matches_finite_diff_relu_pool():
    rng = np.random.default_rng(3)
    img = rng.normal(size=(8, 8))
    kernels = rng.normal(size=(2, 3, 3))

    def loss_fn(kr):
        preact = conv_valid_forward(img, kr)
        relu = np.maximum(preact, 0.0)
        pooled, _ = maxpool_forward(relu, 2)
        return float(np.sum(pooled))

    preact = conv_valid_forward(img, kernels)
    relu = np.maximum(preact, 0.0)
    pooled, argmax = maxpool_forward(relu, 2)
    d_pooled = np.ones_like(pooled)
    d_relu = maxpool_backward(d_pooled, argmax, relu.shape, 2)
    d_preact = d_relu * (preact > 0)
    g = conv_valid_filter_grad(img, d_preact)
    assert np.allclose(g, _fd_grad(loss_fn, kernels.copy()), atol=1e-5)


# -- Stage B: ConvEconomy (filter-level gradient-as-currency) -----------------

from sprout.conv import ConvEconomy


def test_conv_economy_forward_feat_dim_and_inactive_zeros():
    ce = ConvEconomy(k_max=4, kh=3, kw=3, k_init=2, seed=0)   # 2 of 4 active
    h = w = 8
    oh = h - 3 + 1                      # 6 ; pool 2 -> 3
    assert ce.feat_dim(h, w) == 4 * 3 * 3
    feat, cache = ce.forward(np.random.default_rng(0).normal(size=(h, w)))
    assert feat.shape == (4 * 3 * 3,)
    # the two inactive slots (zeroed kernels) contribute exactly zero features
    per = 3 * 3
    assert np.allclose(feat[2 * per:], 0.0)


def test_conv_economy_backward_grad_check_through_economy():
    rng = np.random.default_rng(1)
    ce = ConvEconomy(k_max=2, kh=3, kw=3, seed=1)
    img = rng.normal(size=(8, 8))

    def loss_fn(theta):
        ce.theta = theta
        feat, _ = ce.forward(img)
        return float(feat.sum())                     # d_feat = ones

    feat, cache = ce.forward(img)
    g = ce.backward(np.ones_like(feat), cache)
    # finite-difference check straight through the economy's forward/backward
    fd = np.zeros_like(ce.theta)
    base = ce.theta.copy()
    eps = 1e-6
    it = np.nditer(base, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        p = base.copy(); p[idx] += eps
        m = base.copy(); m[idx] -= eps
        fd[idx] = (loss_fn(p) - loss_fn(m)) / (2 * eps)
        it.iternext()
    ce.theta = base
    assert np.allclose(g, fd, atol=1e-5)


def test_conv_economy_meters_ema():
    ce = ConvEconomy(k_max=2, kh=3, kw=3, seed=0, beta_g=0.5)
    ce.M[:] = 0.0
    g = np.ones((2, 3, 3))
    ce.update_meters(g)                              # ||g_k|| = 3 (sqrt(9))
    assert np.allclose(ce.M, 0.5 * 0.0 + 0.5 * 3.0)


def test_conv_economy_confidence_rewards_loadbearing_settled():
    ce = ConvEconomy(k_max=3, kh=3, kw=3, seed=0, conf_gain=2.0, conf_alpha=0.5)
    # loads via kernel norms: filter 0 big, filter 2 small
    ce.theta[0] = 1.0; ce.theta[1] = 0.5; ce.theta[2] = 0.1
    ce.M[:] = np.array([0.1, 1.0, 1.0])             # filter 0 low demand (settled)
    ce.Mbar_demand = None
    for _ in range(200):
        ce.update_confidence()
    assert ce.conf[0] > 1.0                          # load-bearing AND settled
    assert ce.conf[2] == 0.0                         # below-average load -> imp 0


def test_conv_economy_gated_update_confident_barely_moves():
    ce = ConvEconomy(k_max=2, kh=3, kw=3, seed=0)
    ce.theta[:] = 1.0
    ce.conf[0] = 9.0; ce.conf[1] = 0.0
    g = np.ones((2, 3, 3))
    before = ce.theta.copy()
    ce.gated_update(g, eta=1.0)
    moved0 = np.abs(before[0] - ce.theta[0]).mean()
    moved1 = np.abs(before[1] - ce.theta[1]).mean()
    assert moved0 < moved1 / 5                       # eta/(1+9)=0.1 vs 1.0


def test_conv_economy_prune_drops_lowest_utility_keeps_kmin():
    ce = ConvEconomy(k_max=4, kh=3, kw=3, k_init=4, seed=0, prune_grace=0)
    ce.age[:] = 100
    # make filter 3 inert: tiny norm + tiny demand -> low utility
    ce.theta[0] = 1.0; ce.theta[1] = 1.0; ce.theta[2] = 1.0; ce.theta[3] = 0.01
    ce.M[:] = np.array([1.0, 1.0, 1.0, 0.001])
    pruned = ce.prune(floor=0.5, lam=1.0, k_min=2)
    assert 3 in pruned and ce.active[3] == False
    assert ce.n_active >= 2


def test_conv_economy_prune_protects_high_demand_newborn():
    ce = ConvEconomy(k_max=2, kh=3, kw=3, k_init=2, seed=0, prune_grace=0)
    ce.age[:] = 100
    ce.theta[0] = 1.0; ce.theta[1] = 0.05          # filter 1 low load (newborn-ish)
    ce.M[:] = np.array([1.0, 5.0])                 # but very high demand
    pruned = ce.prune(floor=0.5, lam=1.0, k_min=1)
    assert 1 not in pruned                          # high demand protects it


def test_conv_economy_grow_random_fills_inactive_slot():
    ce = ConvEconomy(k_max=4, kh=3, kw=3, k_init=2, seed=0)
    assert ce.n_active == 2
    born = ce.grow(mode="random", k_max=4)
    assert len(born) == 1 and ce.active[born[0]]
    assert ce.n_active == 3
    assert ce.age[born[0]] == 0 and np.linalg.norm(ce.theta[born[0]]) > 0


def test_conv_economy_grow_split_clones_highest_demand():
    ce = ConvEconomy(k_max=3, kh=3, kw=3, k_init=2, seed=0)
    ce.theta[0] = 0.7; ce.theta[1] = 0.2
    ce.M[:] = np.array([5.0, 0.1, 0.0])            # filter 0 highest demand
    born = ce.grow(mode="split", k_max=3)
    assert len(born) == 1
    # the clone resembles filter 0 (the highest-demand parent), not filter 1
    assert np.linalg.norm(ce.theta[born[0]] - 0.7) < np.linalg.norm(ce.theta[born[0]] - 0.2)


def test_prune_redundant_drops_duplicate_keeps_distinct():
    ce = ConvEconomy(k_max=4, kh=3, kw=3, k_init=4, seed=0)
    base = np.array([[1.0, 0, -1], [1, 0, -1], [1, 0, -1]])   # vertical edge
    ce.theta[0] = base
    ce.theta[1] = base * 0.9 + 1e-4                            # near-duplicate of 0
    ce.theta[2] = base.T                                       # horizontal (distinct)
    ce.theta[3] = np.array([[0, 1.0, 0], [1, -4, 1], [0, 1, 0]])  # blob (distinct)
    ce.M[:] = 1.0
    pruned = ce.prune_redundant(threshold=0.9, k_min=1)
    assert 1 in pruned and 0 not in pruned        # keep the higher-norm duplicate
    assert 2 not in pruned and 3 not in pruned    # distinct filters survive
    assert ce.n_active == 3


def test_prune_redundant_respects_k_min_and_distinct_bank():
    ce = ConvEconomy(k_max=3, kh=3, kw=3, k_init=3, seed=1)
    # three genuinely distinct filters -> nothing pruned
    ce.theta[0] = np.array([[1.0, 0, -1], [1, 0, -1], [1, 0, -1]])
    ce.theta[1] = ce.theta[0].T
    ce.theta[2] = np.array([[0, 1.0, 0], [1, -4, 1], [0, 1, 0]])
    ce.M[:] = 1.0
    assert ce.prune_redundant(threshold=0.9, k_min=1) == []
    assert ce.n_active == 3
