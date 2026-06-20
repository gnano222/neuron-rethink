"""A single live SPROUT training session for the educational web UI.

The browser drives training: it POSTs points+labels to ``start``, then repeatedly
POSTs ``step`` (advance N steps) and renders the returned snapshot, so the polling
loop IS the training cadence (pause = stop polling). One global session is enough
for a local single-user tool; a lock guards the FastAPI threadpool.

The model is the promoted plain SPROUT architecture (phasic-startle-k4): a sparse,
self-rewiring net with 2D confidence, phasic prune/grow at settledness plateaus,
and the startle alarm — no convolution. Topology/density are selectable.
"""
from __future__ import annotations

import threading
from collections import Counter

import numpy as np

from sprout.network import build_graph, init_weights
from sprout.train import Config, Trainer, accuracy
from webapp.snapshot import decision_grid, network_snapshot

DOMAIN = (-1.0, 1.0)          # drawn points + boundary grid live in [-1, 1]^2

# selectable network sizes (input/output pinned to the 2D / 2-class task)
TOPOLOGIES = {
    "small": (2, 8, 8, 2),
    "medium": (2, 12, 12, 2),
    "large": (2, 16, 16, 16, 2),     # the promoted w16 default
}


def _make_config(layers, density):
    """The promoted phasic-startle-k4 architecture at a chosen size (see
    evals.spec._sparse): 2D confidence + phasic prune/grow + sleep + startle."""
    return Config(
        eta_base=0.02, grad_currency=True, enable_confidence=True,
        enable_prune=True, enable_grow=True, gamma_dec=0.001, t_struct=200,
        phasic_structure=True, startle=True, grow_demand_k=4,
        init_layers=tuple(layers), init_density=density)


class TrainingSession:
    def __init__(self):
        self.lock = threading.Lock()
        self._clear()

    def _clear(self):
        self.trainer = self.net = self.X = self.y = None
        self.params = None
        self.loss_ema = None
        self._events_sent = 0

    # -- lifecycle -----------------------------------------------------------
    def start(self, points, labels, *, size="medium", density=0.4, seed=0):
        X = np.asarray(points, dtype=float)
        y = np.asarray(labels, dtype=int)
        if X.ndim != 2 or X.shape[1] != 2 or len(X) < 2:
            raise ValueError("need >= 2 points, each [x, y]")
        layers = TOPOLOGIES.get(size, TOPOLOGIES["medium"])
        with self.lock:
            self.X, self.y = X, y
            self.params = {"size": size, "density": density, "seed": seed}
            self._build(layers, density, seed)

    def restart(self):
        """Re-initialise the net on the SAME dots with a fresh seed (a new self-
        organisation of the same problem)."""
        if self.params is None:
            raise RuntimeError("nothing to restart; call start first")
        with self.lock:
            self.params["seed"] += 1
            self._build(TOPOLOGIES.get(self.params["size"], TOPOLOGIES["medium"]),
                        self.params["density"], self.params["seed"])

    def _build(self, layers, density, seed):
        cfg = _make_config(layers, density)
        net = build_graph(layers, density, seed=seed)
        init_weights(net, seed=seed)
        self.net = net
        self.trainer = Trainer(cfg, net, self.X, self.y, seed=seed)
        self.loss_ema = None
        self._events_sent = 0

    # -- stepping ------------------------------------------------------------
    def step(self, n=20):
        if self.trainer is None:
            raise RuntimeError("call start first")
        with self.lock:
            for _ in range(max(1, int(n))):
                loss = self.trainer.step()
                self.loss_ema = loss if self.loss_ema is None else \
                    0.98 * self.loss_ema + 0.02 * loss

    # -- read-out ------------------------------------------------------------
    def snapshot(self, grid_res=36):
        if self.trainer is None:
            raise RuntimeError("call start first")
        with self.lock:
            net = self.net
            new_events = self.trainer.events[self._events_sent:]
            self._events_sent = len(self.trainer.events)
            counts = Counter(e["type"] for e in new_events)
            if counts.get("startle"):
                phase = "startle"
            elif counts.get("sleep"):
                phase = "consolidating"
            else:
                phase = "settled" if self.trainer.settled else "learning"
            return {
                "step": self.trainer.step_idx,
                "accuracy": accuracy(net, self.X, self.y),
                "loss": float(self.loss_ema) if self.loss_ema is not None else None,
                "synapses": len(net.synapses),
                "neurons": len(net.neurons),
                "graph": network_snapshot(net),
                "boundary": decision_grid(net, grid_res, DOMAIN),
                "domain": list(DOMAIN),
                "phase": phase,
                "events": dict(counts),
                "size": self.params["size"],
            }


SESSION = TrainingSession()
