import numpy as np

from sprout.data import generate_blobs, generate_spirals


def test_blobs_shape_and_labels():
    X, y = generate_blobs(n=200, seed=0)
    assert X.shape == (200, 2)
    assert y.shape == (200,)
    assert set(np.unique(y).tolist()) == {0, 1}


def test_blobs_roughly_balanced():
    X, y = generate_blobs(n=400, seed=1)
    frac = y.mean()
    assert 0.4 < frac < 0.6


def test_blobs_are_linearly_separable_ish():
    # Two gaussian blobs: class means should be well separated relative to spread.
    X, y = generate_blobs(n=400, seed=2)
    m0 = X[y == 0].mean(axis=0)
    m1 = X[y == 1].mean(axis=0)
    centre_dist = np.linalg.norm(m0 - m1)
    spread = X.std(axis=0).mean()
    assert centre_dist > spread  # separated, not on top of each other


def test_blobs_deterministic():
    X1, y1 = generate_blobs(n=100, seed=7)
    X2, y2 = generate_blobs(n=100, seed=7)
    assert np.allclose(X1, X2)
    assert np.array_equal(y1, y2)


def test_spirals_shape_and_labels():
    X, y = generate_spirals(n=300, seed=0)
    assert X.shape == (300, 2)
    assert y.shape == (300,)
    assert set(np.unique(y).tolist()) == {0, 1}


def _best_linear_accuracy(X, y):
    """Accuracy of the best straight-line classifier, found by scanning
    every projection axis and picking the optimal threshold on each."""
    best = 0.0
    for angle in np.linspace(0.0, np.pi, 60, endpoint=False):
        axis = np.array([np.cos(angle), np.sin(angle)])
        proj = X @ axis
        order = np.argsort(proj)
        p, yy = proj[order], y[order]
        # optimal threshold = best split point along this axis
        for thresh in (p[:-1] + p[1:]) / 2.0:
            pred = (p > thresh).astype(int)
            acc = max((pred == yy).mean(), (1 - pred == yy).mean())
            best = max(best, acc)
    return best


def test_spirals_not_linearly_separable():
    # The whole point of spirals: even the best straight line can't split them.
    X, y = generate_spirals(n=400, seed=3)
    assert _best_linear_accuracy(X, y) < 0.85


def test_blobs_are_linearly_separable_by_contrast():
    # Sanity check on the metric: blobs *are* separable by a line.
    X, y = generate_blobs(n=400, seed=3)
    assert _best_linear_accuracy(X, y) > 0.95


def test_spirals_radius_grows_with_angle():
    # Each arm is a real spiral: radius is monotone-ish in the parameter,
    # so the max radius is meaningfully larger than the min radius.
    X, y = generate_spirals(n=400, seed=4)
    r = np.linalg.norm(X, axis=1)
    assert r.max() > 3 * (r.min() + 1e-9)


def test_data_is_normalised_ish():
    # Inputs should be on a sane scale for a tiny net (roughly unit-ish).
    X, _ = generate_spirals(n=400, seed=5)
    assert np.abs(X).max() < 5.0
