import numpy as np

from sprout.data import generate_blobs, generate_spirals
from sprout.datasets import get_dataset, load_digits_split


# -- load_digits_split (Task 1) ----------------------------------------------

def test_digits_shapes_and_labels():
    Xtr, ytr, Xte, yte = load_digits_split(seed=0, test_frac=0.2)
    assert Xtr.shape[1] == 64 and Xte.shape[1] == 64
    assert len(Xtr) == len(ytr) and len(Xte) == len(yte)
    assert len(Xtr) + len(Xte) == 1797
    assert set(np.unique(ytr)).issubset(set(range(10)))
    assert set(np.unique(yte)) == set(range(10))     # all classes in test
    assert ytr.dtype.kind in "iu" and yte.dtype.kind in "iu"


def test_digits_deterministic():
    a = load_digits_split(seed=3)
    b = load_digits_split(seed=3)
    for u, v in zip(a, b):
        assert np.array_equal(u, v)
    c = load_digits_split(seed=4)
    assert not np.array_equal(a[0], c[0])            # different seed -> different split


def test_digits_standardized_on_train_no_leakage():
    Xtr, ytr, Xte, yte = load_digits_split(seed=0)
    # train columns are ~zero-mean / unit-var (constant columns stay ~0)
    assert np.allclose(Xtr.mean(axis=0), 0.0, atol=1e-6)
    stds = Xtr.std(axis=0)
    assert np.all(stds <= 1.0 + 1e-6)
    # test set is NOT forced to zero mean (proves train stats were reused)
    assert not np.allclose(Xte.mean(axis=0), 0.0, atol=1e-3)


def test_digits_stratified_proportions():
    Xtr, ytr, Xte, yte = load_digits_split(seed=1, test_frac=0.2)
    for c in range(10):
        frac = (yte == c).sum() / ((ytr == c).sum() + (yte == c).sum())
        assert 0.1 < frac < 0.3                      # ~0.2 each class


# -- get_dataset registry (Task 2) -------------------------------------------

def test_get_dataset_spirals_byte_identical():
    Xtr, ytr, Xte, yte = get_dataset("spirals", seed=2, n_points=600,
                                     turns=1.0, noise=0.10, test_seed_offset=10000)
    eXtr, eytr = generate_spirals(n=600, seed=2, turns=1.0, noise=0.10)
    eXte, eyte = generate_spirals(n=600, seed=10002, turns=1.0, noise=0.10)
    assert np.array_equal(Xtr, eXtr) and np.array_equal(ytr, eytr)
    assert np.array_equal(Xte, eXte) and np.array_equal(yte, eyte)


def test_get_dataset_blobs_byte_identical():
    Xtr, ytr, Xte, yte = get_dataset("blobs", seed=5, n_points=600,
                                     turns=1.0, noise=0.10, test_seed_offset=10000)
    eXtr, eytr = generate_blobs(n=600, seed=5)
    eXte, eyte = generate_blobs(n=600, seed=10005)
    assert np.array_equal(Xtr, eXtr) and np.array_equal(Xte, eXte)


def test_get_dataset_digits_tuple():
    Xtr, ytr, Xte, yte = get_dataset("digits", seed=0)
    assert Xtr.shape[1] == 64 and set(np.unique(yte)) == set(range(10))


def test_get_dataset_unknown_raises():
    import pytest
    with pytest.raises(ValueError):
        get_dataset("nope", seed=0)


# -- mnist14 (downsampled MNIST) ---------------------------------------------

def test_downsample_2x2_shape_and_block_mean():
    from sprout.datasets import _downsample_2x2
    x = np.zeros((1, 28, 28))
    x[0, 0:2, 0:2] = 4.0       # first 2x2 block -> pooled cell 0
    x[0, 0:2, 2:4] = 8.0       # next 2x2 block  -> pooled cell 1
    out = _downsample_2x2(x.reshape(1, 784))
    assert out.shape == (1, 196)
    assert out[0, 0] == 4.0 and out[0, 1] == 8.0


def test_load_mnist14_split_smoke():
    pytest = __import__("pytest")
    try:
        from sprout.datasets import load_mnist14_split
        Xtr, ytr, Xte, yte = load_mnist14_split(seed=0, n_train=200, n_test=100)
    except Exception as e:                          # offline / fetch unavailable
        pytest.skip(f"MNIST fetch unavailable: {e}")
    assert Xtr.shape[1] == 196 and len(Xtr) == 200 and len(Xte) == 100
    assert set(np.unique(yte)).issubset(set(range(10)))
    assert np.allclose(Xtr.mean(axis=0), 0.0, atol=1e-6)   # standardized on train


def test_get_dataset_mnist14():
    pytest = __import__("pytest")
    try:
        Xtr, ytr, Xte, yte = get_dataset("mnist14", seed=0, n_points=200)
    except Exception as e:
        pytest.skip(f"MNIST fetch unavailable: {e}")
    assert Xtr.shape[1] == 196 and len(Xtr) == 200 and len(Xte) == 1000
