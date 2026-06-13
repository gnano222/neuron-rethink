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
