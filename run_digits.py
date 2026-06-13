"""Stage-1 smoke test: train the SPROUT gradient-as-currency architecture on
scikit-learn 8x8 digits (10 classes) and report accuracy + compute cost.

    python run_digits.py --steps 40000 --seed 0 --layers 64,64,32,10

Mirrors the promoted ``phasic-startle-k4`` eval config (phasic structure + sleep
consolidation + startle), so the numbers here track the eval baseline. No viz —
just train and print a mobile-readable report.
"""

from __future__ import annotations

import argparse
import time

from sprout.datasets import load_digits_split
from sprout.network import build_graph, init_weights
from sprout.train import Config, Trainer, accuracy


def _digits_config() -> Config:
    """The promoted phasic-startle-k4 architecture (matches the eval baseline)."""
    return Config(
        eta_base=0.02, enable_confidence=True, enable_prune=True, enable_grow=True,
        gamma_dec=0.001, t_struct=200, phasic_structure=True, startle=True,
        grow_demand_k=4,
    )


def train_digits(steps=40000, seed=0, layers=(64, 64, 32, 10), density=0.4):
    """Train on digits and return a dict of accuracy + compute metrics."""
    Xtr, ytr, Xte, yte = load_digits_split(seed=seed)
    net = build_graph(list(layers), density=density, seed=seed)
    init_weights(net, seed=seed)
    cfg = _digits_config()
    tr = Trainer(cfg, net, Xtr, ytr, seed=seed)

    edge_steps = 0.0
    t0 = time.perf_counter()
    for _ in range(steps):
        tr.step(record=False)
        edge_steps += len(net.synapses)
    wall = time.perf_counter() - t0

    return {
        "test_acc": accuracy(net, Xte, yte),
        "train_acc": accuracy(net, Xtr, ytr),
        "synapses": len(net.synapses),
        "edge_steps": edge_steps,
        "wall_time": wall,
        "avg_live_edges": edge_steps / steps if steps else 0.0,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="SPROUT on 8x8 digits (smoke test)")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", default="64,64,32,10")
    ap.add_argument("--density", type=float, default=0.4)
    args = ap.parse_args(argv)
    layers = tuple(int(x) for x in args.layers.split(","))
    print(f"training {layers} @ density {args.density} for {args.steps} steps ...")
    r = train_digits(steps=args.steps, seed=args.seed, layers=layers,
                     density=args.density)
    print(f"  test acc      : {r['test_acc']:.4f}")
    print(f"  train acc     : {r['train_acc']:.4f}")
    print(f"  synapses      : {r['synapses']}")
    print(f"  avg live edges: {r['avg_live_edges']:.1f}")
    print(f"  edge-steps    : {r['edge_steps']:.3e}")
    print(f"  wall time     : {r['wall_time']:.1f}s "
          f"({1000 * r['wall_time'] / args.steps:.3f} ms/step)")


if __name__ == "__main__":
    main()
