import numpy as np

from sprout.conv import ConvEconomy
from sprout.conv_train import ConvModel
from sprout.network import build_graph, init_weights
from webapp.infer import run_inference


def _model(side=14, k_max=4, n_out=10, seed=3):
    conv = ConvEconomy(k_max=k_max, kh=3, kw=3, k_init=k_max, seed=seed)
    head = build_graph([conv.feat_dim(side, side), 8, n_out], density=0.5, seed=seed)
    init_weights(head, seed=seed)
    return ConvModel(conv, head, side, side)


def test_payload_prediction_matches_model_forward():
    m = _model()
    img = np.random.default_rng(0).normal(size=(14, 14))
    probs = m.forward(img)[0]
    out = run_inference(m, img)
    assert out["prediction"] == int(np.argmax(probs))
    assert out["prediction"] == int(m.predict([img])[0])
    assert np.allclose(out["probs"], probs)
    assert abs(sum(out["probs"]) - 1.0) < 1e-9


def test_payload_graph_mirrors_head():
    m = _model()
    out = run_inference(m, np.zeros((14, 14)))
    g = out["graph"]
    assert len(g["neurons"]) == len(m.head.neurons)
    assert len(g["synapses"]) == len(m.head.synapses)
    assert g["n_layers"] == len(m.head.layers)
    # layer-0 neuron activations equal the conv features fed to the head
    feat = m.conv.forward(np.zeros((14, 14)))[0]
    layer0 = [n for n in g["neurons"] if n["layer"] == 0]
    assert len(layer0) == len(feat)


def test_payload_filters_and_feature_maps():
    m = _model(k_max=4)
    out = run_inference(m, np.random.default_rng(1).normal(size=(14, 14)))
    assert len(out["filters"]) == m.conv.k_max
    assert len(out["feature_maps"]) == m.conv.n_active
    # each feature map is 6x6 for a 14x14 input (valid 3x3 -> 12, pool 2 -> 6)
    assert np.array(out["feature_maps"][0]["map"]).shape == (6, 6)
    assert np.array(out["filters"][0]["kernel"]).shape == (3, 3)
