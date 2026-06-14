"""Quick single-run smoke test: train the SPROUT phasic-startle-k4 architecture
on a registry dataset and report accuracy + compute cost. The eval harness
(evaluate.py) is the multi-seed comparison tool; this is the fast one-off check.
Defaults to the priority dataset, mnist14.

    python run_dataset.py --dataset mnist14 --steps 40000 --layers 196,64,10
    python run_dataset.py --dataset digits  --steps 40000 --layers 64,64,32,10

Mirrors the promoted ``phasic-startle-k4`` eval config (phasic structure + sleep
consolidation + startle), so the numbers track the eval baseline. No viz — just
train and print a mobile-readable report.
"""

from __future__ import annotations

import argparse
import time

from sprout.datasets import get_dataset
from sprout.network import build_graph, init_weights
from sprout.train import Config, Trainer, accuracy


def _sparse_config() -> Config:
    """The promoted phasic-startle-k4 architecture (matches the eval baseline)."""
    return Config(
        eta_base=0.02, enable_confidence=True, enable_prune=True, enable_grow=True,
        gamma_dec=0.001, t_struct=200, phasic_structure=True, startle=True,
        grow_demand_k=4,
    )


def train_dataset(dataset="mnist14", steps=40000, seed=0, layers=(196, 64, 10),
                  density=0.25, n_points=6000):
    """Train on a registry dataset and return accuracy + compute metrics.

    ``n_points`` is the train-sample count for large/fixed datasets (mnist14);
    it is ignored by the bundled 8x8 ``digits`` set and used as the point count
    for generated ``spirals``/``blobs``.
    """
    Xtr, ytr, Xte, yte = get_dataset(dataset, seed, n_points=n_points)
    net = build_graph(list(layers), density=density, seed=seed)
    init_weights(net, seed=seed)
    tr = Trainer(_sparse_config(), net, Xtr, ytr, seed=seed)

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
    ap = argparse.ArgumentParser(description="SPROUT single-run smoke test")
    ap.add_argument("--dataset", default="mnist14",
                    choices=["mnist14", "digits", "spirals", "blobs"])
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", default="196,64,10")
    ap.add_argument("--density", type=float, default=0.25)
    ap.add_argument("--points", type=int, default=6000,
                    help="train samples (mnist14) / point count (spirals, blobs)")
    args = ap.parse_args(argv)
    layers = tuple(int(x) for x in args.layers.split(","))
    print(f"training {args.dataset} {layers} @ density {args.density} "
          f"for {args.steps} steps ...")
    r = train_dataset(dataset=args.dataset, steps=args.steps, seed=args.seed,
                      layers=layers, density=args.density, n_points=args.points)
    print(f"  test acc      : {r['test_acc']:.4f}")
    print(f"  train acc     : {r['train_acc']:.4f}")
    print(f"  synapses      : {r['synapses']}")
    print(f"  avg live edges: {r['avg_live_edges']:.1f}")
    print(f"  edge-steps    : {r['edge_steps']:.3e}")
    print(f"  wall time     : {r['wall_time']:.1f}s "
          f"({1000 * r['wall_time'] / args.steps:.3f} ms/step)")


if __name__ == "__main__":
    main()
