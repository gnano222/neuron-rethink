"""Run the real Conv-SPROUT forward pass and package everything the UI draws.

The single source of truth is ``ConvModel.forward`` (the same gradient-checked code
the model was trained with). This module only *reads* the resulting state — filter
kernels, per-filter feature maps, and the head network's per-neuron activations and
per-synapse weight/confidence — into a JSON-friendly dict. No model logic here.
"""
from __future__ import annotations

import numpy as np


def run_inference(model, image14) -> dict:
    """``model`` a ConvModel, ``image14`` a standardized 14x14 array. Returns the
    /infer payload: prediction, probs, the input, filters, feature maps, and the
    full live head graph with activations + confidences."""
    img = np.asarray(image14, dtype=float)
    probs, feat, _cache = model.forward(img)   # populates head + conv state
    conv, head = model.conv, model.head

    # per-filter geometry (matches ConvEconomy.forward / feat_dim)
    oh, ow = model.h - conv.kh + 1, model.w - conv.kw + 1
    poh, pow_ = oh // conv.pool, ow // conv.pool
    fmaps = np.asarray(feat, dtype=float).reshape(conv.k_max, poh, pow_)

    filters, feature_maps = [], []
    for slot in range(conv.k_max):
        active = bool(conv.active[slot])
        filters.append({"slot": slot, "active": active,
                        "kernel": conv.theta[slot].tolist() if active else None})
        if active:
            feature_maps.append({"slot": slot, "map": fmaps[slot].tolist()})

    neurons = [{"id": n.id, "layer": n.layer, "x": float(n.pos[0]),
                "y": float(n.pos[1]), "act": float(n.activation)}
               for n in head.neurons]
    synapses = [{"pre": pre, "post": post, "w": float(s.weight),
                 "conf": float(s.confidence)}
                for (pre, post), s in head.synapses.items()]

    return {
        "prediction": int(np.argmax(probs)),
        "probs": [float(p) for p in probs],
        "input_14x14": img.tolist(),
        "filters": filters,
        "feature_maps": feature_maps,
        "graph": {"neurons": neurons, "synapses": synapses,
                  "n_layers": len(head.layers)},
    }
