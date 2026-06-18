import numpy as np

from sprout.conv_features import filter_bank, conv_features


def test_hand_bank_count_shape_norm():
    bank = filter_bank("hand")
    assert len(bank) == 6
    for k in bank:
        assert k.shape == (3, 3)
        assert abs(np.linalg.norm(k) - 1.0) < 1e-9


def test_random_bank_deterministic_and_zero_mean():
    a = filter_bank("random", seed=0)
    b = filter_bank("random", seed=0)
    assert len(a) == 6 and all(np.allclose(x, y) for x, y in zip(a, b))
    c = filter_bank("random", seed=1)
    assert not np.allclose(a[0], c[0])
    for k in a:
        assert abs(k.mean()) < 1e-9 and abs(np.linalg.norm(k) - 1.0) < 1e-9


def test_unknown_bank_raises():
    import pytest
    with pytest.raises(ValueError):
        filter_bank("nope")


def test_conv_features_shape_14x14_hand():
    imgs = np.random.default_rng(0).normal(size=(5, 14, 14))
    feats = conv_features(imgs, filter_bank("hand"), pool=2, nonlin="relu")
    # valid 3x3 -> 12x12, 2x2 pool -> 6x6=36, x6 filters = 216
    assert feats.shape == (5, 216)


def test_relu_nonneg():
    imgs = np.random.default_rng(1).normal(size=(3, 14, 14))
    feats = conv_features(imgs, filter_bank("random", seed=0), pool=2, nonlin="relu")
    assert (feats >= 0).all()


def test_maxpool_translation_invariance():
    # a bright pixel anywhere inside a single 2x2 pool window, after an identity
    # (center-tap) filter, gives the SAME pooled value -> position-invariant.
    bank = [np.array([[0, 0, 0], [0, 1.0, 0], [0, 0, 0]])]
    base = np.zeros((1, 14, 14)); base[0, 5, 5] = 3.0
    shift = np.zeros((1, 14, 14)); shift[0, 5, 6] = 3.0   # +1 col, same pool cell
    fb = conv_features(base, bank, pool=2, nonlin="relu")
    fs = conv_features(shift, bank, pool=2, nonlin="relu")
    assert np.allclose(fb, fs)
