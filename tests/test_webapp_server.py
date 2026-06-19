import numpy as np
from fastapi.testclient import TestClient

from sprout.conv import ConvEconomy
from sprout.conv_train import ConvModel
from sprout.network import build_graph, init_weights
from webapp.serialize import Scaler, save_model


def _fixture_model(path, side=14, k_max=3, n_out=10, seed=0):
    conv = ConvEconomy(k_max=k_max, kh=3, kw=3, k_init=k_max, seed=seed)
    head = build_graph([conv.feat_dim(side, side), 8, n_out], density=0.5, seed=seed)
    init_weights(head, seed=seed)
    model = ConvModel(conv, head, side, side)
    scaler = Scaler(mu=np.zeros(196), sigma=np.ones(196))
    meta = {"side": side, "kh": 3, "kw": 3, "pool": 2, "value_scale": 255.0,
            "classes": list(range(n_out)), "test_acc": 0.5}
    save_model(str(path), model, scaler, meta)


def _client(tmp_path, monkeypatch):
    p = tmp_path / "fixture.pkl"
    _fixture_model(p)
    monkeypatch.setenv("SPROUT_MODEL", str(p))
    from webapp import server
    server._state.clear()                  # drop any cached model from another test
    return TestClient(server.app)


def test_index_serves_html(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    r = client.get("/")
    assert r.status_code == 200
    assert "<canvas" in r.text.lower()


def test_infer_returns_valid_payload(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    canvas = np.zeros((140, 140))
    canvas[30:110, 60:80] = 1.0                       # a vertical stroke
    r = client.post("/infer", json={"pixels": canvas.tolist(), "size": 140})
    assert r.status_code == 200
    d = r.json()
    assert 0 <= d["prediction"] <= 9
    assert len(d["probs"]) == 10 and abs(sum(d["probs"]) - 1.0) < 1e-6
    assert np.array(d["input_14x14"]).shape == (14, 14)
    assert d["graph"]["neurons"] and d["graph"]["synapses"]
    assert "n_active_filters" in d["model_meta"]


def test_graph_returns_resting_topology(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    r = client.get("/graph")
    assert r.status_code == 200
    d = r.json()
    assert d["graph"]["neurons"] and d["graph"]["synapses"]
    assert all(n["act"] == 0.0 for n in d["graph"]["neurons"])
    assert "n_active_filters" in d["model_meta"]


def test_infer_rejects_non_square(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    r = client.post("/infer", json={"pixels": [[0.0, 0.0, 0.0]], "size": 3})
    assert r.status_code == 400
