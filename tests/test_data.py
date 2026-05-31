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


# -- radial-band spirals (concentric continual-learning regime) --------------

def test_spirals_always_centred_on_origin():
    # Two arms offset by pi are point-symmetric, so the cloud mean is the origin.
    X, _ = generate_spirals(n=600, seed=5)
    assert np.allclose(X.mean(axis=0), [0.0, 0.0], atol=0.2)


def test_spirals_inner_band_stays_near_origin():
    # The inner task: a small annular spiral whose points all sit at small radius.
    X, _ = generate_spirals(n=600, seed=6, r_lo=0.15, r_hi=0.55, noise=0.05)
    r = np.linalg.norm(X, axis=1)
    assert r.max() < 0.7          # radius 0.55 + a little noise
    assert r.min() < 0.25         # reaches in toward the centre


def test_spirals_outer_band_is_an_annulus():
    # The outer task: every point sits in a ring well away from the origin, so
    # the inner and outer tasks are disjoint by radius.
    X, _ = generate_spirals(n=600, seed=7, r_lo=0.65, r_hi=1.05, noise=0.05)
    r = np.linalg.norm(X, axis=1)
    assert r.min() > 0.5          # never enters the inner region (gap at ~0.6)
    assert r.max() < 1.2


def test_spirals_default_call_unchanged():
    # Backward compat: explicit default band equals the bare call (the new
    # params must not perturb the RNG draw order).
    a, ya = generate_spirals(n=300, seed=8)
    b, yb = generate_spirals(n=300, seed=8, r_lo=0.2, r_hi=1.0)
    assert np.allclose(a, b)
    assert np.array_equal(ya, yb)
