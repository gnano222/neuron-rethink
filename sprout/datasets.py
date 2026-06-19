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


def _shift_zero_fill(img, dr, dc):
    """Translate a 2D image by ``(dr, dc)`` with ZERO fill (no wrap-around).
    Positive ``dr`` shifts down, positive ``dc`` shifts right; content that falls
    off an edge is discarded and the vacated region is zero."""
    h, w = img.shape
    out = np.zeros_like(img)
    r0s, r1s, c0s, c1s = max(0, -dr), h - max(0, dr), max(0, -dc), w - max(0, dc)
    r0d, r1d, c0d, c1d = max(0, dr), h - max(0, -dr), max(0, dc), w - max(0, -dc)
    out[r0d:r1d, c0d:c1d] = img[r0s:r1s, c0s:c1s]
    return out


def augment_shift(X_flat, side, max_shift=2, n_aug=4, seed=0, include_original=True):
    """Expand flat ``(N, side*side)`` images ``n_aug``-fold with random zero-filled
    translations. Each image yields (optionally) the original plus random shifts in
    ``[-max_shift, max_shift]`` per axis. Returns ``(N*n_aug, side*side)`` with the
    ``n_aug`` variants of image i kept consecutive (so labels = ``np.repeat(y,
    n_aug)``). Pure data augmentation — the network/economy are untouched."""
    rng = np.random.default_rng(seed)
    imgs = np.asarray(X_flat, dtype=float).reshape(-1, side, side)
    out = []
    for img in imgs:
        variants = [img] if include_original else []
        while len(variants) < n_aug:
            dr = int(rng.integers(-max_shift, max_shift + 1))
            dc = int(rng.integers(-max_shift, max_shift + 1))
            variants.append(_shift_zero_fill(img, dr, dc))
        out.extend(v.reshape(-1) for v in variants)
    return np.array(out)


def load_mnist_aug_split(seed: int = 0, n_train: int = 12000, n_test: int = 1000,
                         downsample: bool = True, max_shift: int = 2, n_aug: int = 4):
    """MNIST with TRAIN-only random-shift augmentation, standardized on the
    augmented-train stats; the TEST set is clean (un-augmented). The training set
    expands ``n_aug``-fold (original + random zero-filled shifts). Pure data: tests
    whether translation augmentation lifts the MLP, which has no built-in shift
    invariance. Augmentation happens in pixel space, before standardization.

    CAVEAT: STATIC expansion multiplies the train set ``n_aug``-fold, so at a fixed
    step budget each sample is seen ``1/n_aug`` as often (measured: severe under-
    training). Pair it with ``n_aug``x the steps, or prefer the matched-epoch
    ON-THE-FLY path (``Config.augment_shift_max``), which the eval variants use."""
    from sklearn.datasets import fetch_openml
    data = fetch_openml("mnist_784", version=1, as_frame=False,
                        parser="liac-arff", cache=True)
    X = np.asarray(data.data, dtype=float)
    side = 28
    if downsample:
        X = _downsample_2x2(X)
        side = 14
    y = np.asarray(data.target, dtype=int)
    Xtr, ytr, Xte, yte = _stratified_subsample_split(X, y, n_train, n_test, seed)
    Xtr = augment_shift(Xtr, side, max_shift=max_shift, n_aug=n_aug, seed=seed)
    ytr = np.repeat(ytr, n_aug)
    Xtr, Xte = _standardize_on_train(Xtr, Xte)
    return Xtr, ytr, Xte, yte


def load_mnist_split(seed: int = 0, n_train: int = 12000, n_test: int = 1000,
                     downsample: bool = True):
    """MNIST (fetched via OpenML, cached): a seeded class-balanced subsample,
    standardized on TRAIN stats.

    ``downsample=True`` (the DEFAULT) 2x2-pools 28x28 -> 14x14 (196 features) —
    the canonical SPROUT MNIST, since at a fixed sparse edge budget a 14x14
    thumbnail the net can fully cover beats a 784 image it can only glimpse (see
    docs/findings-2026-06-14...). ``downsample=False`` keeps the full 784 features
    (the ``mnist-full`` dataset). Sparse nets keep edges/compute modest even at
    784 input; only a *dense* 784 net would need the vectorized backend.
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


def mnist_conv_transform(X_flat, side, bank_kind="hand", pool=2, nonlin="relu"):
    """Flat ``(N, side*side)`` images -> ``(N, n_filters*pooled)`` fixed-conv
    features. Pure preprocessing (the bank is fixed at seed 0, never trained), so
    the network/economy are untouched. See sprout/conv_features.py."""
    from sprout.conv_features import filter_bank, conv_features
    imgs = np.asarray(X_flat, dtype=float).reshape(-1, side, side)
    return conv_features(imgs, filter_bank(bank_kind, seed=0), pool=pool, nonlin=nonlin)


def load_mnist_conv_split(seed: int = 0, n_train: int = 12000, n_test: int = 1000,
                          downsample: bool = True, bank_kind: str = "hand",
                          pool: int = 2, nonlin: str = "relu"):
    """MNIST passed through a FIXED conv feature front-end, then standardized on
    TRAIN stats. ``downsample`` 2x2-pools 28->14 first (side=14, else 28). The
    Phase-1 measurement dataset: does translation-invariant input break the ~0.93
    14x14 ceiling? See docs/superpowers/specs/2026-06-16-conv-weight-sharing-design.md.
    """
    from sklearn.datasets import fetch_openml
    data = fetch_openml("mnist_784", version=1, as_frame=False,
                        parser="liac-arff", cache=True)
    X = np.asarray(data.data, dtype=float)                    # (70000, 784)
    side = 28
    if downsample:
        X = _downsample_2x2(X)                                # (70000, 196)
        side = 14
    y = np.asarray(data.target, dtype=int)
    Xtr, ytr, Xte, yte = _stratified_subsample_split(X, y, n_train, n_test, seed)
    Ftr = mnist_conv_transform(Xtr, side, bank_kind, pool, nonlin)
    Fte = mnist_conv_transform(Xte, side, bank_kind, pool, nonlin)
    Ftr, Fte = _standardize_on_train(Ftr, Fte)
    return Ftr, ytr, Fte, yte


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
    if name == "mnist":                                 # 14x14 (the default MNIST)
        return load_mnist_split(seed=seed, n_train=n_points, n_test=1000,
                                downsample=True)
    if name == "mnist-full":                            # full 784
        return load_mnist_split(seed=seed, n_train=n_points, n_test=1000,
                                downsample=False)
    if name == "mnist-aug":                             # 14x14 + train shift-augment
        return load_mnist_aug_split(seed=seed, n_train=n_points, n_test=1000,
                                    downsample=True)
    if name == "mnist-conv":                            # 14x14 -> hand-filter conv
        return load_mnist_conv_split(seed=seed, n_train=n_points, n_test=1000,
                                     downsample=True, bank_kind="hand")
    if name == "mnist-conv-rand":                       # 14x14 -> random-filter conv
        return load_mnist_conv_split(seed=seed, n_train=n_points, n_test=1000,
                                     downsample=True, bank_kind="random")
    if name == "mnist-full-conv":                       # 28x28 -> hand-filter conv
        return load_mnist_conv_split(seed=seed, n_train=n_points, n_test=1000,
                                     downsample=False, bank_kind="hand")
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
