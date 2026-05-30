"""Visualization (§9) - the payoff.

Renders the two object-lists into a multi-panel figure:

  * Main panel  - the network graph. Neurons are dots (size/brightness from
    activation); synapses are lines (thickness from |weight|, colour from
    confidence: blue = unsure/fast-learning -> red = confident/frozen).
  * Side panels - accuracy vs step, synapse count vs step, and the 2D decision
    boundary with the data overlaid.

Because we run headless, ``render_frame`` writes a PNG; :func:`make_gif`
stitches a sequence of PNGs into an animation. The same figure layout works
for a live ``matplotlib.FuncAnimation`` loop if run interactively.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

# blue (unsure / fast) -> red (confident / frozen)
_CONF_CMAP = plt.get_cmap("coolwarm")


def _grid_predictions(net, X, res=70, margin=0.4):
    lo = X.min(axis=0) - margin
    hi = X.max(axis=0) + margin
    xs = np.linspace(lo[0], hi[0], res)
    ys = np.linspace(lo[1], hi[1], res)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    preds = np.array([net.forward(p)[1] for p in pts])  # P(class 1)
    return gx, gy, preds.reshape(gx.shape), (lo, hi)


def _draw_network(ax, net, edge_norm, edge_cmap, edge_attr):
    ax.set_title("network graph", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    # synapses as a LineCollection: thickness ~ |weight|, colour ~ edge_attr
    segs, widths, colors = [], [], []
    max_w = max((abs(s.weight) for s in net.synapses.values()), default=1.0) or 1.0
    for (pre, post), syn in net.synapses.items():
        p0 = net.neurons[pre].pos
        p1 = net.neurons[post].pos
        segs.append([p0, p1])
        widths.append(0.3 + 3.5 * abs(syn.weight) / max_w)
        colors.append(edge_cmap(edge_norm(getattr(syn, edge_attr))))
    if segs:
        lc = LineCollection(segs, linewidths=widths, colors=colors, alpha=0.75, zorder=1)
        ax.add_collection(lc)

    # neurons as dots: size/brightness ~ activation
    xs = [n.pos[0] for n in net.neurons]
    ys = [n.pos[1] for n in net.neurons]
    acts = np.array([n.activation for n in net.neurons])
    a_max = acts.max() if acts.max() > 0 else 1.0
    sizes = 40 + 260 * (acts / a_max)
    bright = 0.25 + 0.75 * (acts / a_max)
    ax.scatter(xs, ys, s=sizes, c=bright, cmap="Greys", vmin=0, vmax=1,
               edgecolors="black", linewidths=0.6, zorder=2)

    ax.set_xlim(min(xs) - 0.5, max(xs) + 0.5)
    ax.set_ylim(min(ys) - 1.0, max(ys) + 1.0)
    ax.set_aspect("equal")


def _edge_style(edge_mode, net, conf_vmax=3.0):
    """Map an edge-colouring mode to ``(norm, cmap, synapse-attr, label)``.

      * ``"confidence"`` (default) - blue=fast/unsure -> red=confident/frozen.
      * ``"demand"``     - gradient-as-currency view: how hard the loss is still
        pushing each wire (``grad_mag``); dark=settled -> bright=being pushed.
      * ``"eligibility"`` - legacy v1 three-factor "glow" on co-active wires.

    Inferno-coloured modes use an adaptive max so the signal is always visible.
    """
    if edge_mode == "demand":
        m_max = max((s.grad_mag for s in net.synapses.values()), default=0.0)
        return (Normalize(0.0, max(m_max, 1e-9)), plt.get_cmap("inferno"),
                "grad_mag", "gradient demand (dark=settled → bright=being pushed)")
    if edge_mode == "eligibility":
        e_max = max((s.eligibility for s in net.synapses.values()), default=0.0)
        return (Normalize(0.0, max(e_max, 1e-6)), plt.get_cmap("inferno"),
                "eligibility", "eligibility (dark=quiet → bright=co-active)")
    return (Normalize(0.0, conf_vmax), _CONF_CMAP,
            "confidence", "confidence (blue=fast → red=frozen)")


def render_frame(net, trainer, X, y, path, step=None, conf_vmax=3.0,
                 edge_mode="confidence"):
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.5, 1, 1], height_ratios=[1, 1])

    edge_norm, edge_cmap, edge_attr, cb_label = _edge_style(edge_mode, net, conf_vmax)

    ax_net = fig.add_subplot(gs[:, 0])
    _draw_network(ax_net, net, edge_norm, edge_cmap, edge_attr)

    sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=edge_norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_net, fraction=0.04, pad=0.02)
    cb.set_label(cb_label, fontsize=8)

    # accuracy
    ax_acc = fig.add_subplot(gs[0, 1])
    h = trainer.history
    if h["accuracy"]:
        ax_acc.plot(h["rec_step"], h["accuracy"], color="green")
    ax_acc.set_title("accuracy", fontsize=10)
    ax_acc.set_ylim(0, 1.02)
    ax_acc.grid(alpha=0.3)

    # synapse count
    ax_syn = fig.add_subplot(gs[1, 1])
    ax_syn.plot(h["step"], h["synapse_count"], color="purple")
    ax_syn.set_title("synapse count", fontsize=10)
    ax_syn.grid(alpha=0.3)

    # decision boundary
    ax_db = fig.add_subplot(gs[:, 2])
    gx, gy, gp, _ = _grid_predictions(net, X)
    ax_db.contourf(gx, gy, gp, levels=20, cmap="RdBu_r", alpha=0.7, vmin=0, vmax=1)
    ax_db.scatter(X[y == 0, 0], X[y == 0, 1], s=8, c="navy", edgecolors="white", linewidths=0.2)
    ax_db.scatter(X[y == 1, 0], X[y == 1, 1], s=8, c="darkred", edgecolors="white", linewidths=0.2)
    ax_db.set_title("decision boundary", fontsize=10)
    ax_db.set_xticks([])
    ax_db.set_yticks([])
    ax_db.set_aspect("equal")

    title = f"SPROUT — step {step if step is not None else trainer.step_idx}"
    acc = h["accuracy"][-1] if h["accuracy"] else None
    if acc is not None:
        title += f"   acc={acc:.2f}   synapses={len(net.synapses)}"
    fig.suptitle(title, fontsize=12)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


def make_gif(frame_paths, out_path, duration=0.25):
    from PIL import Image
    frames = [Image.open(p) for p in frame_paths]
    if not frames:
        return None
    frames[0].save(
        out_path, save_all=True, append_images=frames[1:],
        duration=int(duration * 1000), loop=0,
    )
    return out_path
