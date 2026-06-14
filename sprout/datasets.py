"""Dataset registry for SPROUT.

Unifies *generated* toy datasets (spirals/blobs, reproduced byte-for-byte from
``sprout.data`` so the eval cache and pinned baselines are undisturbed) and
*fixed* datasets (scikit-learn 8x8 digits) behind one ``get_dataset`` interface
returning ``(X_tr, y_tr, X_te, y_te)``. New fixed datasets (e.g. MNIST) plug in
here as one extra loader branch; the eval harness never changes.
"""

from __future__ import annotations

import numpy as np

from sprout.data import generate_blobs, generate_spirals

_EPS = 1e-8


def _stratified_split(X, y, test_frac, seed):
    """Deterministic per-class shuffle + split (no sklearn.model_selection dep)."""
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_frac))
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())
    train_idx = np.array(sorted(train_idx))
    test_idx = np.array(sorted(test_idx))
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def _standardize_on_train(Xtr, Xte):
    """Z-score both splits using TRAIN statistics only (no leakage; the eps guard
    handles constant columns, e.g. always-blank corner pixels)."""
    mu = Xtr.mean(axis=0)
    sigma = Xtr.std(axis=0)
    return (Xtr - mu) / (sigma + _EPS), (Xte - mu) / (sigma + _EPS)


def _downsample_2x2(X):
    """Flatten-784 MNIST images -> flatten-196 by 2x2 mean pooling (28->14)."""
    n = X.shape[0]
    return X.reshape(n, 14, 2, 14, 2).mean(axis=(2, 4)).reshape(n, 196)


def _stratified_subsample_split(X, y, n_train, n_test, seed):
    """Class-balanced, deterministic, disjoint train/test subsample of a large
    fixed dataset (per-class: first n_train/K to train, next n_test/K to test)."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per_tr = n_train // len(classes)
    per_te = n_test // len(classes)
    tr_idx, te_idx = [], []
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        tr_idx.extend(idx[:per_tr].tolist())
        te_idx.extend(idx[per_tr:per_tr + per_te].tolist())
    tr_idx = np.array(sorted(tr_idx))
    te_idx = np.array(sorted(te_idx))
    return X[tr_idx], y[tr_idx], X[te_idx], y[te_idx]


def load_mnist_split(seed: int = 0, n_train: int = 12000, n_test: int = 1000,
                     downsample: bool = False):
    """MNIST (fetched via OpenML, cached): a seeded class-balanced subsample,
    standardized on TRAIN stats. ``downsample`` 2x2-pools 28x28 -> 14x14 (196
    features); otherwise the full 784 features are used. Sparse nets keep the
    edge budget (hence compute) modest even at 784 input, so full-res is tractable
    in the per-synapse loop; only a *dense* 784 net would need a vectorized backend.
    """
    from sklearn.datasets import fetch_openml
    data = fetch_openml("mnist_784", version=1, as_frame=False,
                        parser="liac-arff", cache=True)
    X = np.asarray(data.data, dtype=float)                    # (70000, 784)
    if downsample:
        X = _downsample_2x2(X)                                # (70000, 196)
    y = np.asarray(data.target, dtype=int)                    # labels 0..9
    Xtr, ytr, Xte, yte = _stratified_subsample_split(X, y, n_train, n_test, seed)
    Xtr, Xte = _standardize_on_train(Xtr, Xte)
    return Xtr, ytr, Xte, yte


def load_mnist14_split(seed: int = 0, n_train: int = 3000, n_test: int = 1000):
    """MNIST 2x2-pooled to 14x14 = 196 features (a harder task than 8x8 digits at
    a tractable edge scale). Thin wrapper over ``load_mnist_split``."""
    return load_mnist_split(seed, n_train, n_test, downsample=True)


def load_digits_split(seed: int = 0, test_frac: float = 0.2):
    """scikit-learn 8x8 digits: seeded stratified split, standardized on TRAIN
    statistics only (no leakage; the eps guard handles constant corner pixels)."""
    from sklearn.datasets import load_digits
    data = load_digits()
    X = np.asarray(data.data, dtype=float)        # (1797, 64), pixels 0..16
    y = np.asarray(data.target, dtype=int)        # labels 0..9
    Xtr, ytr, Xte, yte = _stratified_split(X, y, test_frac, seed)
    Xtr, Xte = _standardize_on_train(Xtr, Xte)
    return Xtr, ytr, Xte, yte


def get_dataset(name, seed, *, n_points=600, turns=1.0, noise=0.10,
                test_seed_offset=10000):
    """Return ``(X_tr, y_tr, X_te, y_te)`` for a named dataset.

    Generated datasets draw train at ``seed`` and test at
    ``seed + test_seed_offset`` with the exact args the runner used before, so
    spirals/blobs are byte-identical (eval cache + pinned baselines safe). Fixed
    datasets (digits) ignore the generator knobs.
    """
    if name == "digits":
        return load_digits_split(seed=seed)
    if name == "mnist14":
        return load_mnist14_split(seed=seed, n_train=n_points, n_test=1000)
    if name == "mnist":
        return load_mnist_split(seed=seed, n_train=n_points, n_test=1000)
    if name == "blobs":
        Xtr, ytr = generate_blobs(n=n_points, seed=seed)
        Xte, yte = generate_blobs(n=n_points, seed=seed + test_seed_offset)
        return Xtr, ytr, Xte, yte
    if name == "spirals":
        Xtr, ytr = generate_spirals(n=n_points, seed=seed, turns=turns, noise=noise)
        Xte, yte = generate_spirals(n=n_points, seed=seed + test_seed_offset,
                                    turns=turns, noise=noise)
        return Xtr, ytr, Xte, yte
    raise ValueError(f"unknown dataset {name!r}")
