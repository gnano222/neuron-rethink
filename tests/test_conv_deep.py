import numpy as np

from sprout.conv import conv_valid_forward
from sprout.conv_deep import (ConvLayer, DeepConvModel, DeepConvTrainer,
                              conv_mc_forward, conv_mc_filter_grad,
                              conv_mc_input_grad)
from sprout.network import build_graph, init_weights
from sprout.train import Config


def _num_grad(loss_fn, x, eps=1e-6):
    """Central finite-difference gradient of scalar ``loss_fn()`` w.r.t. array x."""
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps; fp = loss_fn()
        x[idx] = old - eps; fm = loss_fn()
        x[idx] = old
        g[idx] = (fp - fm) / (2 * eps)
        it.iternext()
    return g


# -- multi-channel conv math (the new stacking primitives) -------------------

def test_mc_forward_reduces_to_single_channel():
    rng = np.random.default_rng(0)
    img = rng.normal(size=(5, 5))
    kernels = rng.normal(size=(3, 3, 3))               # K=3 single-channel 3x3
    mc = conv_mc_forward(img[None], kernels[:, None])  # add C=1 axis
    sc = conv_valid_forward(img, kernels)
    assert np.allclose(mc, sc)


def test_mc_forward_shape():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, 7, 6))                     # C=4, H=7, W=6
    kernels = rng.normal(size=(5, 4, 3, 3))            # K=5, C=4
    out = conv_mc_forward(x, kernels)
    assert out.shape == (5, 5, 4)                      # oh=5, ow=4


def test_mc_filter_grad_matches_numeric():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(3, 6, 5))
    kernels = rng.normal(size=(4, 3, 3, 3))
    r = rng.normal(size=(4, 4, 3))                     # upstream grad on preact

    def loss():
        return float((conv_mc_forward(x, kernels) * r).sum())

    analytic = conv_mc_filter_grad(x, r)
    numeric = _num_grad(loss, kernels)
    assert np.allclose(analytic, numeric, atol=1e-5)


def test_mc_input_grad_matches_numeric():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(3, 6, 5))
    kernels = rng.normal(size=(4, 3, 3, 3))
    r = rng.normal(size=(4, 4, 3))

    def loss():
        return float((conv_mc_forward(x, kernels) * r).sum())

    analytic = conv_mc_input_grad(r, kernels)
    numeric = _num_grad(loss, x)
    assert np.allclose(analytic, numeric, atol=1e-5)


# -- ConvLayer (multi-channel currency layer) --------------------------------

def test_convlayer_forward_shape_and_feat_dim():
    layer = ConvLayer(k=4, c_in=3, seed=0)
    x = np.random.default_rng(0).normal(size=(3, 13, 13))
    pooled, _ = layer.forward(x)
    assert pooled.shape == (4, 5, 5)                # 13->11 conv, 11->5 pool
    assert layer.feat_dim(13, 13) == 4 * 5 * 5


def test_convlayer_backward_shapes_and_optional_input_grad():
    layer = ConvLayer(k=3, c_in=2, seed=2)
    x = np.random.default_rng(2).normal(size=(2, 8, 8))
    pooled, cache = layer.forward(x)
    d_pooled = np.random.default_rng(3).normal(size=pooled.shape)
    g, d_x = layer.backward(d_pooled, cache, need_input_grad=True)
    assert g.shape == (3, 2, 3, 3) and d_x.shape == (2, 8, 8)
    _, dx_none = layer.backward(d_pooled, cache, need_input_grad=False)
    assert dx_none is None


def test_convlayer_full_backward_gradient_check():
    rng = np.random.default_rng(5)
    layer = ConvLayer(k=2, c_in=2, pool=2, seed=5)
    x = rng.normal(size=(2, 7, 7))
    pooled, cache = layer.forward(x)
    r = rng.normal(size=pooled.shape)

    def loss():
        p, _ = layer.forward(x)
        return float((p * r).sum())

    g, d_x = layer.backward(r, cache)
    assert np.allclose(g, _num_grad(loss, layer.theta), atol=1e-5)   # filter grad
    assert np.allclose(d_x, _num_grad(loss, x), atol=1e-5)           # input grad


def test_convlayer_meters_and_confidence_brake():
    layer = ConvLayer(k=2, c_in=1, seed=1)
    g = np.ones((2, 1, 3, 3))
    layer.update_meters(g)
    assert layer.M[0] > 0.0 and layer.Svec.shape == (2, 1, 3, 3)
    before = layer.theta.copy()
    layer.gated_update(g, eta=0.1)
    free_step = np.abs(layer.theta - before).max()
    layer.conf[:] = 9.0                            # confident -> eta/(1+9)=0.01
    t2 = layer.theta.copy()
    layer.gated_update(g, eta=0.1)
    braked_step = np.abs(layer.theta - t2).max()
    assert braked_step < free_step                 # confidence slows learning
    layer.conf[:] = 0.0                            # natural confidence stays bounded
    for _ in range(50):
        layer.update_meters(g)
        layer.update_confidence()
    assert np.all((layer.conf >= 0.0) & (layer.conf <= layer.c_max))


# -- DeepConvModel (two stacked layers + sparse head) ------------------------

def _two_layer_model(h=10, w=10, seed=0):
    l1 = ConvLayer(k=2, c_in=1, seed=1)            # 10->8 conv, 8->4 pool
    l2 = ConvLayer(k=3, c_in=2, seed=2)            # 4->2 conv, 2->1 pool -> (3,1,1)
    head = build_graph([3, 5, 4], density=1.0, seed=seed)
    init_weights(head, seed=seed)
    return DeepConvModel([l1, l2], head, h, w), l1, l2


def test_deepconvmodel_feat_dim_and_forward():
    model, l1, l2 = _two_layer_model()
    assert model.feat_dim() == 3                    # (3,1,1)
    probs, feat, caches = model.forward(np.random.default_rng(0).normal(size=(10, 10)))
    assert feat.shape == (3,) and probs.shape == (4,) and len(caches) == 2


def test_deepconvmodel_stack_gradient_check():
    rng = np.random.default_rng(7)
    model, l1, l2 = _two_layer_model()
    img = rng.normal(size=(10, 10))
    y = 2

    def loss():
        model.forward(img)
        L, _, _ = model.head.backward(y)
        return L

    _, _, caches = model.forward(img)
    _, _, gb = model.head.backward(y)
    grads = model.backward_convs(gb, caches)
    # both layers' filter grads must match finite differences through the WHOLE
    # stack (head delta -> last layer -> input grad -> first layer)
    assert np.allclose(grads[0], _num_grad(loss, l1.theta), atol=1e-4)
    assert np.allclose(grads[1], _num_grad(loss, l2.theta), atol=1e-4)


def test_deepconvtrainer_learns_tiny_task():
    rng = np.random.default_rng(0)
    h = w = 12
    # tiny memorization task: two class-prototype patterns + per-sample noise
    proto = rng.normal(size=(2, h, w))
    X = np.array([proto[i % 2] + 0.3 * rng.normal(size=(h, w)) for i in range(12)])
    y = np.array([i % 2 for i in range(12)])
    l1 = ConvLayer(k=4, c_in=1, seed=1)            # 12->10->5
    l2 = ConvLayer(k=4, c_in=4, seed=2)            # 5->3->1 -> feat 4
    head = build_graph([4, 8, 2], density=1.0, seed=0)
    init_weights(head, seed=0)
    model = DeepConvModel([l1, l2], head, h, w)
    cfg = Config(eta_base=0.05, enable_confidence=True, enable_prune=False,
                 enable_grow=False, phasic_structure=True, startle=False)
    tr = DeepConvTrainer(cfg, model, X, y, seed=0, conv_eta=0.05,
                         conv_eta_schedule="none")
    early = np.mean([tr.step() for _ in range(50)])
    for _ in range(600):
        tr.step()
    late = np.mean([tr.step() for _ in range(50)])
    assert late < 0.6 * early                       # the stack learns end-to-end
    assert tr.accuracy(X, y) >= 0.83
