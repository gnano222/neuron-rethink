"""Run the real Conv-SPROUT forward pass and package everything the UI draws.

The single source of truth is ``ConvModel.forward`` (the same gradient-checked code
the model was trained with). This module only *reads* the resulting state — filter
kernels, per-filter feature maps, and the head network's per-neuron activations and
per-synapse weight/confidence — into a JSON-friendly dict. No model logic here.
"""
from __future__ import annotations

import numpy as np


def _filters_payload(conv):
    """The learned kernels (model-constant, drawing-independent)."""
    return [{"slot": s, "active": bool(conv.active[s]),
             "kernel": conv.theta[s].tolist() if conv.active[s] else None}
            for s in range(conv.k_max)]


def _graph_payload(head, *, resting=False):
    """The full head topology. ``resting`` zeroes activations (no drawing yet)."""
    neurons = [{"id": n.id, "layer": n.layer, "x": float(n.pos[0]),
                "y": float(n.pos[1]),
                "act": 0.0 if resting else float(n.activation)}
               for n in head.neurons]
    synapses = [{"pre": pre, "post": post, "w": float(s.weight),
                 "conf": float(s.confidence)}
                for (pre, post), s in head.synapses.items()]
    return {"neurons": neurons, "synapses": synapses, "n_layers": len(head.layers)}


def resting_payload(model) -> dict:
    """Filters + full head topology with no activation — the at-rest view shown on
    page load and after Clear (every wire visible, nothing fired)."""
    return {"filters": _filters_payload(model.conv),
            "graph": _graph_payload(model.head, resting=True)}


def run_inference(model, image14) -> dict:
    """``model`` a ConvModel, ``image14`` a standardized 14x14 array. Returns the
    /infer payload: prediction, probs, the input, filters, feature maps, and the
    full live head graph with activations + weights/confidences."""
    img = np.asarray(image14, dtype=float)
    probs, feat, _cache = model.forward(img)   # populates head + conv state
    conv, head = model.conv, model.head

    # per-filter geometry (matches ConvEconomy.forward / feat_dim)
    oh, ow = model.h - conv.kh + 1, model.w - conv.kw + 1
    poh, pow_ = oh // conv.pool, ow // conv.pool
    fmaps = np.asarray(feat, dtype=float).reshape(conv.k_max, poh, pow_)

    feature_maps = [{"slot": s, "map": fmaps[s].tolist()}
                    for s in range(conv.k_max) if conv.active[s]]

    return {
        "prediction": int(np.argmax(probs)),
        "probs": [float(p) for p in probs],
        "input_14x14": img.tolist(),
        "filters": _filters_payload(conv),
        "feature_maps": feature_maps,
        "graph": _graph_payload(head),
    }
