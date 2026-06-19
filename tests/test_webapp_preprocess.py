import numpy as np

from webapp.preprocess import _center_in_field, to_model_input
from webapp.serialize import Scaler


def _identity_scaler():
    # mu=0, sigma=1 -> transform is ~identity (x / (1+eps)), so we can inspect pixels
    return Scaler(mu=np.zeros(196), sigma=np.ones(196))


def test_center_in_field_recenters_corner_blob():
    img = np.zeros((280, 280))
    img[10:60, 10:60] = 1.0                       # blob jammed in the top-left
    out = _center_in_field(img)
    total = out.sum()
    com_r = (out.sum(axis=1) @ np.arange(28)) / total
    com_c = (out.sum(axis=0) @ np.arange(28)) / total
    assert abs(com_r - 13.5) < 1.5 and abs(com_c - 13.5) < 1.5   # near field center


def test_output_is_14x14():
    img = np.zeros((140, 140))
    img[40:100, 60:75] = 1.0
    out = to_model_input(img, _identity_scaler())
    assert out.shape == (14, 14)


def test_blank_canvas_standardizes_background():
    scaler = Scaler(mu=np.full(196, 3.0), sigma=np.ones(196))
    out = to_model_input(np.zeros((128, 128)), scaler)
    assert np.allclose(out, scaler.transform(np.zeros(196)).reshape(14, 14))


def test_uses_saved_stats_not_recomputed():
    img = np.zeros((140, 140))
    img[30:110, 55:85] = 1.0
    a = to_model_input(img, Scaler(mu=np.zeros(196), sigma=np.ones(196)))
    b = to_model_input(img, Scaler(mu=np.full(196, 50.0), sigma=np.ones(196)))
    # same drawing, different saved mu -> different model input (stats are applied)
    assert not np.allclose(a, b)


def test_accepts_0_255_scale_like_0_1():
    img01 = np.zeros((140, 140))
    img01[40:100, 50:80] = 1.0
    img255 = img01 * 255.0
    s = _identity_scaler()
    assert np.allclose(to_model_input(img01, s), to_model_input(img255, s))
