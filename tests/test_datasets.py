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


# -- MNIST: `mnist` = 14x14 (default), `mnist-full` = 784 ---------------------

def test_downsample_2x2_shape_and_block_mean():
    from sprout.datasets import _downsample_2x2
    x = np.zeros((1, 28, 28))
    x[0, 0:2, 0:2] = 4.0       # first 2x2 block -> pooled cell 0
    x[0, 0:2, 2:4] = 8.0       # next 2x2 block  -> pooled cell 1
    out = _downsample_2x2(x.reshape(1, 784))
    assert out.shape == (1, 196)
    assert out[0, 0] == 4.0 and out[0, 1] == 8.0


def test_load_mnist_split_defaults_to_14x14():
    pytest = __import__("pytest")
    try:
        from sprout.datasets import load_mnist_split
        Xtr, ytr, Xte, yte = load_mnist_split(seed=0, n_train=200, n_test=100)
    except Exception as e:                          # offline / fetch unavailable
        pytest.skip(f"MNIST fetch unavailable: {e}")
    assert Xtr.shape[1] == 196 and len(Xtr) == 200 and len(Xte) == 100   # 14x14
    assert set(np.unique(yte)).issubset(set(range(10)))
    assert np.allclose(Xtr.mean(axis=0), 0.0, atol=1e-6)   # standardized on train


def test_get_dataset_mnist_is_14x14():
    pytest = __import__("pytest")
    try:
        Xtr, ytr, Xte, yte = get_dataset("mnist", seed=0, n_points=200)
    except Exception as e:
        pytest.skip(f"MNIST fetch unavailable: {e}")
    assert Xtr.shape[1] == 196 and len(Xtr) == 200 and len(Xte) == 1000   # default = 14x14


def test_get_dataset_mnist_full_is_784():
    pytest = __import__("pytest")
    try:
        Xtr, ytr, Xte, yte = get_dataset("mnist-full", seed=0, n_points=200)
    except Exception as e:
        pytest.skip(f"MNIST fetch unavailable: {e}")
    assert Xtr.shape[1] == 784 and len(Xtr) == 200 and len(Xte) == 1000
    assert set(np.unique(yte)).issubset(set(range(10)))


# -- Fashion-MNIST (harder substrate with headroom) --------------------------

def test_get_dataset_fashion_is_14x14():
    pytest = __import__("pytest")
    try:
        Xtr, ytr, Xte, yte = get_dataset("fashion", seed=0, n_points=200)
    except Exception as e:
        pytest.skip(f"Fashion-MNIST fetch unavailable: {e}")
    assert Xtr.shape[1] == 196 and len(Xtr) == 200 and len(Xte) == 1000
    assert set(np.unique(yte)).issubset(set(range(10)))
    assert np.allclose(Xtr.mean(axis=0), 0.0, atol=1e-6)


def test_get_dataset_fashion_full_is_784():
    pytest = __import__("pytest")
    try:
        Xtr, ytr, Xte, yte = get_dataset("fashion-full", seed=0, n_points=200)
    except Exception as e:
        pytest.skip(f"Fashion-MNIST fetch unavailable: {e}")
    assert Xtr.shape[1] == 784 and len(Xtr) == 200 and len(Xte) == 1000


# -- conv front-end datasets (Phase 1) ---------------------------------------

def test_mnist_conv_transform_shape():
    from sprout.datasets import mnist_conv_transform
    X = np.random.default_rng(0).normal(size=(4, 196))     # 4 fake 14x14, flat
    F = mnist_conv_transform(X, side=14, bank_kind="hand", pool=2, nonlin="relu")
    assert F.shape == (4, 216)                              # 6 filters x 6x6


def test_mnist_conv_transform_is_deterministic_fixed_bank():
    from sprout.datasets import mnist_conv_transform
    X = np.random.default_rng(1).normal(size=(3, 196))
    a = mnist_conv_transform(X, side=14, bank_kind="random")
    b = mnist_conv_transform(X, side=14, bank_kind="random")
    assert np.array_equal(a, b)                             # bank fixed (seed 0)


def test_get_dataset_unknown_conv_raises():
    import pytest
    with pytest.raises(ValueError):
        get_dataset("mnist-conv-nope", seed=0)


# -- train-shift augmentation (perf lever #2) --------------------------------

def test_shift_zero_fill_moves_content_and_zero_fills():
    from sprout.datasets import _shift_zero_fill
    img = np.zeros((14, 14)); img[5, 5] = 3.0
    s = _shift_zero_fill(img, 1, -2)              # down 1, left 2
    assert s[6, 3] == 3.0 and s[5, 5] == 0.0


def test_shift_zero_fill_does_not_wrap():
    from sprout.datasets import _shift_zero_fill
    img = np.zeros((14, 14)); img[7, 13] = 5.0   # bright pixel on the right edge
    s = _shift_zero_fill(img, 0, 1)              # shift right -> falls off the edge
    assert s.sum() == 0.0                         # nothing wrapped around
    assert (s[:, 0] == 0.0).all()                # vacated column is zero


def test_augment_shift_shape_and_original_first():
    from sprout.datasets import augment_shift
    X = np.random.default_rng(0).normal(size=(5, 196))
    A = augment_shift(X, side=14, max_shift=2, n_aug=4, seed=0)
    assert A.shape == (20, 196)                   # 5 images x 4 variants
    for i in range(5):                            # variant 0 = untouched original
        assert np.array_equal(A[i * 4], X[i])


def test_augment_shift_deterministic():
    from sprout.datasets import augment_shift
    X = np.random.default_rng(2).normal(size=(4, 196))
    a = augment_shift(X, side=14, seed=7)
    b = augment_shift(X, side=14, seed=7)
    assert np.array_equal(a, b)


def test_get_dataset_mnist_aug_expands_train_clean_test():
    pytest = __import__("pytest")
    try:
        Xtr, ytr, Xte, yte = get_dataset("mnist-aug", seed=0, n_points=200)
    except Exception as e:
        pytest.skip(f"MNIST fetch unavailable: {e}")
    assert Xtr.shape[1] == 196
    assert len(Xtr) == 200 * 4 and len(ytr) == 200 * 4    # 4x augmented train
    assert len(Xte) == 1000                                # clean (un-augmented) test
    assert np.allclose(Xtr.mean(axis=0), 0.0, atol=1e-6)   # standardized on aug-train


def test_get_dataset_mnist_conv_is_216():
    pytest = __import__("pytest")
    try:
        Xtr, ytr, Xte, yte = get_dataset("mnist-conv", seed=0, n_points=200)
    except Exception as e:
        pytest.skip(f"MNIST fetch unavailable: {e}")
    assert Xtr.shape[1] == 216 and len(Xtr) == 200 and len(Xte) == 1000
    assert np.allclose(Xtr.mean(axis=0), 0.0, atol=1e-6)    # standardized on train
