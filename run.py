"""Experiment driver for SPROUT.

Runs the network on a dataset with a chosen set of mechanisms enabled,
following the build order in §10. Produces:

  * a sequence of PNG frames + an animated GIF (output/<name>/...)
  * a metrics.json with the recorded history, structural events, and the
    §11 validation checks.

Usage:
    python run.py --preset step1 --dataset blobs --steps 4000
    python run.py --preset full  --dataset spirals --steps 30000
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from sprout.data import generate_blobs, generate_spirals
from sprout.network import build_graph, init_weights
from sprout.train import Config, Trainer, accuracy
from sprout import viz


# cumulative build-order presets (§10)
PRESETS = {
    "step1": dict(),
    "step2": dict(enable_eligibility=True),
    "step3": dict(enable_eligibility=True, enable_confidence=True),
    "step4": dict(enable_eligibility=True, enable_confidence=True, enable_prune=True),
    "step5": dict(enable_eligibility=True, enable_confidence=True, enable_prune=True,
                  enable_grow=True),
    # homeostasis is opt-in: the spec's weight-rescaling form is unstable with
    # ReLU and the net is stable without it, so "full" leaves it off.
    "step6": dict(enable_eligibility=True, enable_confidence=True, enable_prune=True,
                  enable_grow=True, enable_homeostasis=True),
    "full": dict(enable_eligibility=True, enable_confidence=True, enable_prune=True,
                 enable_grow=True),
}


def make_config(preset, **overrides):
    cfg = Config(**PRESETS[preset])
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def run(preset="step1", dataset="blobs", steps=4000, seed=0,
        layers=(2, 8, 8, 6, 2), density=0.5,
        record_every=100, draw_every=400, out=None, n_points=600,
        cfg_overrides=None, render=True, edge_mode=None,
        turns=1.0, noise=0.10):
    # step2 has no confidence signal yet, so colour edges by eligibility
    if edge_mode is None:
        edge_mode = "eligibility" if preset == "step2" else "confidence"
    out = out or os.path.join("output", f"{preset}_{dataset}")
    os.makedirs(out, exist_ok=True)

    if dataset == "blobs":
        X, y = generate_blobs(n=n_points, seed=seed)
    else:
        X, y = generate_spirals(n=n_points, seed=seed, turns=turns, noise=noise)

    net = build_graph(list(layers), density=density, seed=seed)
    init_weights(net, seed=seed)
    cfg = make_config(preset, **(cfg_overrides or {}))
    trainer = Trainer(cfg, net, X, y, seed=seed)

    frames = []
    for s in range(steps):
        record = (s % record_every == 0) or (s == steps - 1)
        trainer.step(record=record)
        if render and (s % draw_every == 0 or s == steps - 1):
            p = os.path.join(out, f"frame_{len(frames):04d}.png")
            viz.render_frame(net, trainer, X, y, p, step=s, edge_mode=edge_mode)
            frames.append(p)

    final_acc = accuracy(net, X, y)
    if render:
        viz.make_gif(frames, os.path.join(out, "animation.gif"), duration=0.4)

    metrics = summarise(trainer, net, final_acc, cfg)
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=float)

    return trainer, net, metrics, out


def summarise(trainer, net, final_acc, cfg):
    h = trainer.history
    confs = [s.confidence for s in net.synapses.values()]
    elig = [s.eligibility for s in net.synapses.values()]
    n_prune_events = sum(1 for e in trainer.events if e["type"] == "prune")
    n_grow_events = sum(1 for e in trainer.events if e["type"] == "grow")
    syn = h["synapse_count"]
    return {
        "preset_flags": {
            "eligibility": cfg.enable_eligibility,
            "confidence": cfg.enable_confidence,
            "prune": cfg.enable_prune,
            "grow": cfg.enable_grow,
            "homeostasis": cfg.enable_homeostasis,
        },
        "final_accuracy": final_acc,
        "max_accuracy": max(h["accuracy"]) if h["accuracy"] else None,
        "synapse_count_start": syn[0] if syn else None,
        "synapse_count_end": syn[-1] if syn else None,
        "synapse_count_max": max(syn) if syn else None,
        "synapse_count_min": min(syn) if syn else None,
        "confidence_max": max(confs) if confs else 0.0,
        "confidence_mean": float(np.mean(confs)) if confs else 0.0,
        "eligibility_max": max(elig) if elig else 0.0,
        "n_prune_events": n_prune_events,
        "n_grow_events": n_grow_events,
        "n_neurons": net.num_neurons,
    }


def _parse_layers(s):
    return tuple(int(x) for x in s.split(","))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="step1", choices=list(PRESETS))
    ap.add_argument("--dataset", default="blobs", choices=["blobs", "spirals"])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", type=_parse_layers, default=None,
                    help="comma-separated, e.g. 2,10,10,8,2")
    ap.add_argument("--density", type=float, default=0.5)
    ap.add_argument("--eta", type=float, default=None)
    ap.add_argument("--theta-prune", type=float, default=None)
    ap.add_argument("--prune-warmup", type=int, default=None)
    ap.add_argument("--turns", type=float, default=1.0)
    ap.add_argument("--noise", type=float, default=0.10)
    ap.add_argument("--record-every", type=int, default=100)
    ap.add_argument("--draw-every", type=int, default=400)
    ap.add_argument("--points", type=int, default=600)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    # sensible defaults for the harder spirals task
    spirals = args.dataset == "spirals"
    layers = args.layers or ((2, 10, 10, 8, 2) if spirals else (2, 8, 8, 6, 2))
    overrides = {}
    overrides["eta_base"] = args.eta if args.eta is not None else (0.02 if spirals else 0.05)
    if args.theta_prune is not None:
        overrides["theta_prune"] = args.theta_prune
    elif spirals:
        overrides["theta_prune"] = 0.001
    if args.prune_warmup is not None:
        overrides["prune_warmup"] = args.prune_warmup
    elif spirals and args.preset in ("step4", "step5", "step6", "full"):
        overrides["prune_warmup"] = 6000

    _, _, metrics, out = run(
        preset=args.preset, dataset=args.dataset, steps=args.steps, seed=args.seed,
        layers=layers, density=args.density, record_every=args.record_every,
        draw_every=args.draw_every, n_points=args.points, out=args.out,
        render=not args.no_render, turns=args.turns, noise=args.noise,
        cfg_overrides=overrides,
    )
    print(json.dumps(metrics, indent=2, default=float))
    print("artifacts ->", out)
