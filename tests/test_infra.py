"""Infra tests: the gradient-as-currency mechanism is the architecture wired
through run.py presets and the viz edge-colouring.
"""

import os

from run import PRESETS, make_config, DEFAULT_PRESET
from sprout.viz import _edge_style, render_frame
from sprout.network import build_graph, init_weights
from sprout.data import generate_blobs
from sprout.train import Trainer


# -- run.py presets ----------------------------------------------------------

def test_currency_is_the_default_preset():
    assert DEFAULT_PRESET == "currency"
    assert DEFAULT_PRESET in PRESETS


def test_currency_preset_enables_the_currency_stack():
    cfg = make_config("currency")
    assert cfg.enable_confidence and cfg.enable_prune and cfg.enable_grow


def test_core_preset_is_plain_backprop():
    cfg = make_config("core")
    assert not (cfg.enable_confidence or cfg.enable_prune or cfg.enable_grow)


# -- viz edge styling --------------------------------------------------------

def test_edge_style_demand_colours_by_gradient_meter():
    net = build_graph([2, 4, 2], density=1.0, seed=0)
    _, _, attr, label = _edge_style("demand", net)
    assert attr == "grad_mag"
    assert "demand" in label.lower()


def test_edge_style_confidence_is_default():
    net = build_graph([2, 4, 2], density=1.0, seed=0)
    _, _, attr, _ = _edge_style("confidence", net)
    assert attr == "confidence"


def test_edge_style_eligibility_preserved_for_legacy():
    net = build_graph([2, 4, 2], density=1.0, seed=0)
    _, _, attr, _ = _edge_style("eligibility", net)
    assert attr == "eligibility"


def test_render_frame_demand_mode_writes_png(tmp_path):
    net = build_graph([2, 6, 4, 2], density=0.6, seed=0)
    init_weights(net, seed=0)
    X, y = generate_blobs(n=60, seed=0)
    tr = Trainer(make_config("currency", eta_base=0.05), net, X, y, seed=0)
    for s in range(40):
        tr.step(record=(s % 10 == 0))
    p = render_frame(net, tr, X, y, str(tmp_path / "f.png"), step=40, edge_mode="demand")
    assert os.path.exists(p)
