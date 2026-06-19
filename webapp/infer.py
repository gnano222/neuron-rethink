"""Run the real Conv-SPROUT forward pass and package everything the UI draws.

The single source of truth is ``ConvModel.forward`` (the same gradient-checked code
the model was trained with). This module only *reads* the resulting state — filter
kernels, per-filter feature maps, and the head network's per-neuron activations and
per-synapse weight/confidence — into a JSON-friendly dict. No model logic here.
"""
from __future__ import annotations

import numpy as np


def _filters_payload(conv, contrib=None):
    """The learned kernels (model-constant). ``contrib`` (slot -> float) attaches
    each active filter's contribution to the current prediction, when available."""
    return [{"slot": s, "active": bool(conv.active[s]),
             "kernel": conv.theta[s].tolist() if conv.active[s] else None,
             "contribution": (contrib.get(s) if (contrib and conv.active[s]) else None)}
            for s in range(conv.k_max)]


def _logit(head, nid):
    """Pre-softmax score of output neuron ``nid`` from the head's current hidden
    activations (softmax would saturate to ~1.0 on a confident digit and hide all
    contribution, so we attribute in logit space)."""
    z = head.neurons[nid].bias
    for pre in head.incoming[nid]:
        z += head.synapses[(pre, nid)].weight * head.neurons[pre].activation
    return z


def _filter_contributions(model, feat, pred):
    """How much each filter drives THIS prediction, by occlusion: zero the filter's
    features, re-run the head, and measure the drop in the predicted digit's logit.
    Positive = the filter supports the prediction; ~0 = irrelevant; negative = it
    argues against. Cheap (one head forward per active filter). The head must hold
    the full forward on entry (for the baseline); it is restored on exit."""
    head, conv = model.head, model.conv
    per = ((model.h - conv.kh + 1) // conv.pool) * ((model.w - conv.kw + 1) // conv.pool)
    pred_nid = head.layers[-1][pred]
    base = _logit(head, pred_nid)
    feat = np.asarray(feat, dtype=float)
    contrib = {}
    for slot in range(conv.k_max):
        if not conv.active[slot]:
            continue
        occluded = feat.copy()
        occluded[slot * per:(slot + 1) * per] = 0.0
        model.head.forward(occluded)
        contrib[slot] = float(base - _logit(head, pred_nid))
    model.head.forward(feat)                          # restore true activations
    return contrib


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
    graph = _graph_payload(head)                 # snapshot activations before occlusion

    pred = int(np.argmax(probs))
    contrib = _filter_contributions(model, feat, pred)

    return {
        "prediction": pred,
        "probs": [float(p) for p in probs],
        "input_14x14": img.tolist(),
        "filters": _filters_payload(conv, contrib),
        "feature_maps": feature_maps,
        "graph": graph,
    }
