"""Serialize a live SPROUT ``Network`` into JSON the browser can animate.

Three views, all read-only:
  * ``network_snapshot`` - neurons (by layer, with firing rate) + every live
    synapse (weight, confidence, age). This is what "watch the wires evolve" draws.
  * ``decision_grid`` - P(class 1) over a square grid of the input domain, for the
    decision-boundary heatmap behind the dots.
  * metrics (accuracy/synapse counts) are assembled in session.py.

Node activity is the firing-rate EMA ``r_j`` (a stable "how active is this neuron"
signal), NOT the last forward activation — so the boundary-grid forwards below
don't perturb what we display.
"""
from __future__ import annotations

import numpy as np


def network_snapshot(net) -> dict:
    neurons = [{"id": n.id, "layer": n.layer, "rate": float(n.firing_rate)}
               for n in net.neurons]
    synapses = [{"pre": pre, "post": post, "w": float(s.weight),
                 "conf": float(s.confidence), "age": int(s.age)}
                for (pre, post), s in net.synapses.items()]
    return {"neurons": neurons, "synapses": synapses,
            "n_layers": len(net.layers),
            "layer_sizes": [len(layer) for layer in net.layers]}


def decision_grid(net, res=36, domain=(-1.0, 1.0)) -> list:
    """P(class 1) over a ``res`` x ``res`` grid, rows top->bottom (image order)."""
    lo, hi = domain
    xs = np.linspace(lo, hi, res)
    ys = np.linspace(hi, lo, res)          # top row = high y
    grid = []
    for yv in ys:
        row = [float(net.forward(np.array([xv, yv]))[1]) for xv in xs]
        grid.append(row)
    return grid
