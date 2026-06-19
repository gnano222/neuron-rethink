"""Persist a trained Conv-SPROUT model for the web explorer.

A ``ConvModel`` is plain Python objects + numpy arrays (the filter economy and the
sparse head ``Network``), so it pickles directly. We bundle it with the input
``Scaler`` (the per-pixel train mean/std a drawn digit must be standardized by) and
small ``meta`` so the server is self-contained.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np

# Matches sprout.datasets._standardize_on_train so a drawn digit is z-scored
# exactly as the training data was.
_EPS = 1e-8


@dataclass
class Scaler:
    """Per-pixel z-score using the TRAIN statistics captured at export time."""

    mu: np.ndarray
    sigma: np.ndarray

    def transform(self, flat):
        flat = np.asarray(flat, dtype=float)
        return (flat - self.mu) / (self.sigma + _EPS)


def save_model(path, model, scaler: Scaler, meta: dict) -> None:
    with open(path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "meta": meta}, f)


def load_model(path):
    """Return ``(model, scaler, meta)`` from a file written by :func:`save_model`."""
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["model"], d["scaler"], d["meta"]
