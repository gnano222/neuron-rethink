import numpy as np

from webapp.session import TrainingSession


def _two_blobs(n=40, seed=0):
    """Linearly separable: class 0 left, class 1 right, in [-1, 1]^2."""
    rng = np.random.default_rng(seed)
    a = rng.normal([-0.6, 0.0], 0.12, size=(n, 2))
    b = rng.normal([0.6, 0.0], 0.12, size=(n, 2))
    X = np.clip(np.vstack([a, b]), -1, 1)
    y = np.array([0] * n + [1] * n)
    return X.tolist(), y.tolist()


def test_start_snapshot_shapes():
    s = TrainingSession()
    pts, lab = _two_blobs()
    s.start(pts, lab, size="small")
    snap = s.snapshot(grid_res=20)
    assert snap["step"] == 0
    assert len(snap["graph"]["neurons"]) == sum(snap["graph"]["layer_sizes"])
    assert snap["graph"]["synapses"]
    assert len(snap["boundary"]) == 20 and len(snap["boundary"][0]) == 20
    assert all(0.0 <= v <= 1.0 for row in snap["boundary"] for v in row)
    assert 0.0 <= snap["accuracy"] <= 1.0


def test_start_requires_two_points():
    s = TrainingSession()
    import pytest
    with pytest.raises(ValueError):
        s.start([[0.0, 0.0]], [0])


def test_step_improves_accuracy_on_separable_data():
    s = TrainingSession()
    pts, lab = _two_blobs()
    s.start(pts, lab, size="small", seed=1)
    before = s.snapshot(grid_res=8)["accuracy"]
    s.step(6000)
    after = s.snapshot(grid_res=8)["accuracy"]
    assert after > 0.9            # learns to separate two blobs
    assert after >= before


def test_restart_resets_step_and_bumps_seed():
    s = TrainingSession()
    pts, lab = _two_blobs()
    s.start(pts, lab, size="small", seed=5)
    s.step(500)
    assert s.snapshot(grid_res=8)["step"] == 500
    s.restart()
    assert s.snapshot(grid_res=8)["step"] == 0
    assert s.params["seed"] == 6


def test_phase_and_events_keys_present():
    s = TrainingSession()
    pts, lab = _two_blobs()
    s.start(pts, lab, size="small")
    s.step(50)
    snap = s.snapshot(grid_res=8)
    assert snap["phase"] in {"learning", "settled", "consolidating", "startle"}
    assert isinstance(snap["events"], dict)
