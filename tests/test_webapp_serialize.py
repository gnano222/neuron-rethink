import numpy as np

from sprout.conv import ConvEconomy
from sprout.conv_train import ConvModel
from sprout.network import build_graph, init_weights
from webapp.serialize import Scaler, load_model, save_model


def _tiny_model(side=14, k_max=2, n_out=3, seed=0):
    conv = ConvEconomy(k_max=k_max, kh=3, kw=3, k_init=k_max, seed=seed)
    head = build_graph([conv.feat_dim(side, side), 4, n_out], density=0.5, seed=seed)
    init_weights(head, seed=seed)
    return ConvModel(conv, head, side, side)


def test_save_load_roundtrip_preserves_prediction(tmp_path):
    m = _tiny_model()
    img = np.random.default_rng(1).normal(size=(14, 14))
    before = m.forward(img)[0]

    scaler = Scaler(mu=np.arange(196.0), sigma=np.ones(196))
    meta = {"side": 14, "kh": 3, "kw": 3, "pool": 2, "value_scale": 255.0,
            "classes": [0, 1, 2]}
    p = tmp_path / "m.pkl"
    save_model(str(p), m, scaler, meta)

    m2, scaler2, meta2 = load_model(str(p))
    assert np.allclose(m2.forward(img)[0], before)        # same weights -> same output
    assert meta2 == meta
    assert np.allclose(scaler2.mu, scaler.mu) and np.allclose(scaler2.sigma, scaler.sigma)


def test_scaler_transform_is_zscore_with_eps():
    s = Scaler(mu=np.array([1.0, 2.0]), sigma=np.array([2.0, 0.0]))
    out = s.transform([3.0, 2.0])
    assert np.allclose(out, [(3 - 1) / (2 + 1e-8), 0.0 / (0.0 + 1e-8)])
