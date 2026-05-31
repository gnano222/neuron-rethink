"""Toy 2D datasets for SPROUT (§6).

Two tasks:
  * blobs  - two Gaussian clusters, (almost) linearly separable; for debugging.
  * spirals - two interleaving spirals; the real test that forces the net to use
    its capacity so growth/pruning actually matter.

All generators are deterministic given ``seed`` and return ``(X, y)`` with
``X`` of shape ``(n, 2)`` (float64) and integer labels ``y`` in ``{0, 1}``.
Inputs are kept on a roughly unit scale so the tiny net trains sanely.
"""

from __future__ import annotations

import numpy as np


def generate_blobs(n: int = 800, seed: int = 0, spread: float = 0.5):
    """Two Gaussian blobs with well-separated means."""
    rng = np.random.default_rng(seed)
    n0 = n // 2
    n1 = n - n0
    centres = np.array([[-1.0, -1.0], [1.0, 1.0]])
    X0 = rng.normal(centres[0], spread, size=(n0, 2))
    X1 = rng.normal(centres[1], spread, size=(n1, 2))
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])
    perm = rng.permutation(n)
    return X[perm], y[perm]


def generate_spirals(n: int = 800, seed: int = 0, noise: float = 0.18, turns: float = 1.5,
                     r_lo: float = 0.2, r_hi: float = 1.0):
    """Two interleaving spirals (one per class), offset by pi radians.

    The radius sweeps the band ``[r_lo, r_hi]`` linearly with the spiral
    parameter, so each arm is a genuine spiral and the two classes are not
    linearly separable. The default band ``(0.2, 1.0)`` reproduces the original
    unit-scale spiral byte-for-byte (it is pure arithmetic around the unchanged
    RNG draws). The continual-learning regime uses two *concentric* bands - an
    inner annular spiral (task A) and a disjoint outer one (task B) - so both
    tasks stay centred on the origin (zero-mean, learnable by the tiny net) yet
    occupy separate regions of the plane (jointly valid).
    """
    rng = np.random.default_rng(seed)
    n0 = n // 2
    n1 = n - n0

    def arm(count, phase):
        t = np.linspace(0.0, 1.0, count)
        r = r_lo + (r_hi - r_lo) * t           # radius sweeps the band
        theta = turns * 2.0 * np.pi * t + phase
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        pts = np.stack([x, y], axis=1)
        pts = pts + rng.normal(0.0, noise * r[:, None], size=pts.shape)
        return pts

    X0 = arm(n0, phase=0.0)
    X1 = arm(n1, phase=np.pi)
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])
    perm = rng.permutation(n)
    return X[perm], y[perm]
