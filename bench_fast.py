"""Benchmark the vectorized backend (ArrayNet) vs the object Network on the
per-step WAKE path, across edge counts. Reports ms/step and speedup.

    python bench_fast.py
"""

from __future__ import annotations

import time

import numpy as np

from sprout.network import build_graph, init_weights
from sprout.fast import ArrayNet
from sprout import learning, currency
from sprout.train import Config


def _cfg():
    return Config(eta_base=0.02, enable_confidence=True, gamma_dec=0.001,
                  grow_demand_k=4)


def _obj_wake_step(net, cfg, x, y, t):
    net.forward(x)
    learning.update_firing_rates(net, cfg.beta)
    _, gw, gb = net.backward(int(y))
    currency.update_gradient_meters(net, gw, cfg.beta_g, step_idx=t, lazy=False)
    currency.update_confidence_2d(net, cfg.conf_gain, cfg.conf_alpha, cfg.c_max,
                                  cfg.settled_mode, cfg.conf_k)
    learning.apply_gated_update(net, gw, gb, cfg.eta_base)
    for syn in net.synapses.values():
        syn.age += 1


def _time(fn, n):
    t0 = time.perf_counter()
    for t in range(n):
        fn(t)
    return (time.perf_counter() - t0) / n * 1000.0   # ms/step


def bench(layers, density, steps=200, seed=0):
    cfg = _cfg()
    n_in, n_out = layers[0], layers[-1]
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(128, n_in))
    Y = rng.integers(0, n_out, 128)
    stream = rng.integers(0, 128, size=steps)

    net = build_graph(list(layers), density=density, seed=seed)
    init_weights(net, seed=seed)
    edges = len(net.synapses)
    an = ArrayNet.from_network(net)

    obj_ms = _time(lambda t: _obj_wake_step(net, cfg, X[stream[t]], Y[stream[t]], t), steps)
    arr_ms = _time(lambda t: an.step(X[stream[t]], int(Y[stream[t]]), cfg, t), steps)
    return edges, obj_ms, arr_ms


def main():
    configs = [
        ((2, 16, 16, 16, 2), 0.4),      # spirals w16
        ((64, 64, 32, 10), 0.4),        # digits
        ((196, 64, 10), 0.5),           # mnist14 ~2x budget
        ((784, 64, 10), 0.25),          # full mnist, modest
        ((784, 128, 64, 10), 0.3),      # bigger
        ((784, 256, 128, 10), 0.4),     # large
    ]
    print(f"{'layers':24s} {'edges':>7s} {'obj ms/step':>12s} "
          f"{'arr ms/step':>12s} {'speedup':>8s}")
    print("-" * 68)
    for layers, d in configs:
        edges, obj_ms, arr_ms = bench(layers, d)
        sp = obj_ms / arr_ms if arr_ms else float("inf")
        print(f"{str(layers):24s} {edges:7d} {obj_ms:12.3f} {arr_ms:12.3f} {sp:7.1f}x")


if __name__ == "__main__":
    main()
