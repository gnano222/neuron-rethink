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


def load_digits_split(seed: int = 0, test_frac: float = 0.2):
    """scikit-learn 8x8 digits: seeded stratified split, standardized on TRAIN
    statistics only (no leakage; the eps guard handles constant corner pixels)."""
    from sklearn.datasets import load_digits
    data = load_digits()
    X = np.asarray(data.data, dtype=float)        # (1797, 64), pixels 0..16
    y = np.asarray(data.target, dtype=int)        # labels 0..9
    Xtr, ytr, Xte, yte = _stratified_split(X, y, test_frac, seed)
    mu = Xtr.mean(axis=0)
    sigma = Xtr.std(axis=0)
    Xtr = (Xtr - mu) / (sigma + _EPS)
    Xte = (Xte - mu) / (sigma + _EPS)
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
