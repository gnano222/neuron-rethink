"""Turn a raw drawing canvas into the exact input Conv-SPROUT was trained on.

The model saw MNIST: 28x28 grayscale (white digit on black, values 0..255), 2x2
mean-pooled to 14x14, then per-pixel z-scored on the TRAIN stats. A hand drawing
only gets recognized if it enters the model the same way, so we mirror MNIST's own
normalization: crop to the ink, scale into a 20x20 box (aspect preserved), center
by center-of-mass in a 28x28 field, pool to 14x14, then standardize with the saved
scaler.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from sprout.datasets import _downsample_2x2, _shift_zero_fill

_FIT_BOX = 20      # MNIST scales each digit to fit a 20x20 box ...
_FIELD = 28        # ... centered in a 28x28 field
_INK = 0.05        # treat pixels above this (0..1) as ink


def _to_unit(image):
    """Square 2D array -> float in [0,1] with ink bright. Accepts 0..1 or 0..255."""
    arr = np.asarray(image, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"image must be a square 2D array, got shape {arr.shape}")
    if arr.max() > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _ink_bbox(mask):
    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]
    return rows[0], rows[-1] + 1, cols[0], cols[-1] + 1


def _center_in_field(cropped, fit_box=_FIT_BOX, field=_FIELD):
    """Resize a cropped digit into ``fit_box`` (aspect preserved), paste it centered
    in a ``field`` x ``field`` black image, then shift so its center of mass sits at
    the field center (matches MNIST's centering)."""
    h, w = cropped.shape
    scale = fit_box / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    small = np.asarray(
        Image.fromarray((cropped * 255).astype(np.uint8)).resize((nw, nh), Image.LANCZOS),
        dtype=float) / 255.0

    out = np.zeros((field, field))
    top, left = (field - nh) // 2, (field - nw) // 2
    out[top:top + nh, left:left + nw] = small

    total = out.sum()
    if total > 0:
        com_r = (out.sum(axis=1) @ np.arange(field)) / total
        com_c = (out.sum(axis=0) @ np.arange(field)) / total
        center = (field - 1) / 2.0
        out = _shift_zero_fill(out, int(round(center - com_r)),
                               int(round(center - com_c)))
    return out


def to_model_input(image, scaler, *, value_scale=255.0, ink_threshold=_INK):
    """Drawing canvas (square 2D, ink bright) -> standardized 14x14 ready for
    ``ConvModel.forward``. A blank canvas standardizes to the all-background image."""
    arr = _to_unit(image)
    mask = arr > ink_threshold
    if not mask.any():                                   # blank canvas
        return scaler.transform(np.zeros(_FIELD // 2 * (_FIELD // 2))).reshape(14, 14)

    r0, r1, c0, c1 = _ink_bbox(mask)
    field28 = _center_in_field(arr[r0:r1, c0:c1])
    pooled = _downsample_2x2(field28.reshape(1, _FIELD * _FIELD))[0]   # 196, in [0,1]
    return scaler.transform(pooled * value_scale).reshape(14, 14)
