"""Fixed convolutional feature front-end (Phase 1 of conv/weight-sharing).

A *fixed* (untrained) bank of small filters turns a raw image into translation-
tolerant feature maps, which SPROUT then classifies as if they were the input.
Because the filters never learn, this is pure preprocessing: the network, the
currency economy, and the array backend are untouched. The point is to MEASURE
how much translation invariance buys before building the faithful weight-sharing
version (Phase 2). See docs/superpowers/specs/2026-06-16-conv-weight-sharing-design.md.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

_EPS = 1e-12


def _hand_bank():
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)  # vertical edge
    sy = sx.T                                                          # horizontal edge
    d1 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=float)  # 45 deg
    d2 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype=float)  # 135 deg
    center = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=float)  # blob on
    surround = -center                                                # blob off
    return [sx, sy, d1, d2, center, surround]


def filter_bank(kind="hand", seed=0, size=3):
    """Return a list of ``(size,size)`` l2-normalized kernels.

    ``hand``: 4 oriented edges + 2 center-surround blobs (6 filters; ``size`` must
    be 3 -- the classic V1 detectors). ``random``: 6 zero-mean, unit-norm Gaussian
    kernels (fixed by ``seed``), for a structure-free A/B against the hand bank.
    """
    if kind == "hand":
        if size != 3:
            raise ValueError("hand bank is defined for size=3 only")
        bank = _hand_bank()
    elif kind == "random":
        rng = np.random.default_rng(seed)
        bank = [rng.normal(size=(size, size)) for _ in range(6)]
        bank = [k - k.mean() for k in bank]
    else:
        raise ValueError(f"unknown filter bank kind {kind!r}")
    return [k / (np.linalg.norm(k) + _EPS) for k in bank]


def _nonlin(x, kind):
    if kind == "relu":
        return np.maximum(x, 0.0)
    if kind == "abs":
        return np.abs(x)
    raise ValueError(f"unknown nonlin {kind!r}")


def _maxpool(a, pool):
    """Non-overlapping 2D max-pool over the last two axes (drops a ragged edge)."""
    n, h, w = a.shape
    hc, wc = (h // pool) * pool, (w // pool) * pool
    a = a[:, :hc, :wc].reshape(n, hc // pool, pool, wc // pool, pool)
    return a.max(axis=(2, 4))


def conv_features(images, bank, pool=2, nonlin="relu"):
    """Apply each fixed kernel (valid correlation), the nonlinearity, and max-pool,
    then flatten + concatenate across filters.

    ``images``: ``(N, H, W)``. Returns ``(N, n_filters * pooled_h * pooled_w)``.
    Vectorized over all images at once via a sliding-window view -- a one-time
    precompute, so per-training-step SPROUT compute is unchanged.
    """
    images = np.asarray(images, dtype=float)
    feats = []
    for k in bank:
        kh, kw = k.shape
        win = sliding_window_view(images, (kh, kw), axis=(1, 2))   # (N,oh,ow,kh,kw)
        conv = np.einsum("nijkl,kl->nij", win, k)
        feats.append(_maxpool(_nonlin(conv, nonlin), pool).reshape(len(images), -1))
    return np.concatenate(feats, axis=1)
