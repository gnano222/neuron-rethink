"""Train the promoted Conv-SPROUT model on MNIST-14 and export it for the web app.

Reproduces the `conv-sprout` arm from conv_experiment.py (k_max=12 3x3 filters,
cosine filter consolidation, activation-redundancy prune @0.80, phasic head), then
serializes the model + the train scaler + meta to webapp/model/conv_sprout.pkl.

    .venv/bin/python -m webapp.train_export --steps 20000 --seed 0
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from sprout.conv import ConvEconomy
from sprout.conv_train import ConvModel, ConvTrainer
from sprout.datasets import _downsample_2x2, _stratified_subsample_split
from sprout.network import build_graph, init_weights
from sprout.train import Config
from webapp.serialize import Scaler, save_model

_EPS = 1e-8                              # matches datasets._standardize_on_train
_HERE = os.path.dirname(__file__)
_DEFAULT_OUT = os.path.join(_HERE, "model", "conv_sprout.pkl")


def _load_mnist14(seed, n_train, n_test):
    """Raw MNIST 2x2-pooled to 14x14 (0..255), split, and the TRAIN mean/std.

    We replicate load_mnist_split but KEEP mu/sigma (the scaler a drawn digit must
    be standardized by) instead of discarding them inside _standardize_on_train."""
    from sklearn.datasets import fetch_openml
    data = fetch_openml("mnist_784", version=1, as_frame=False,
                        parser="liac-arff", cache=True)
    X = _downsample_2x2(np.asarray(data.data, dtype=float))      # (70000, 196), 0..255
    y = np.asarray(data.target, dtype=int)
    Xtr, ytr, Xte, yte = _stratified_subsample_split(X, y, n_train, n_test, seed)
    mu, sigma = Xtr.mean(axis=0), Xtr.std(axis=0)
    Xtr_s = (Xtr - mu) / (sigma + _EPS)
    Xte_s = (Xte - mu) / (sigma + _EPS)
    return Xtr_s, ytr, Xte_s, yte, Scaler(mu=mu, sigma=sigma)


def _head_config():
    return Config(eta_base=0.02, enable_confidence=True, enable_prune=True,
                  enable_grow=True, gamma_dec=0.001, t_struct=200,
                  phasic_structure=True, startle=False, grow_demand_k=4,
                  sleep_warmup=2500, sleep_patience=1500)


def build_conv_sprout(side, n_out, seed, total_steps, conv_eta=0.02):
    conv = ConvEconomy(k_max=12, kh=3, kw=3, k_init=12, seed=seed)
    head = build_graph([conv.feat_dim(side, side), 32, n_out], density=0.5, seed=seed)
    init_weights(head, seed=seed)
    model = ConvModel(conv, head, side, side)
    trainer = ConvTrainer(
        _head_config(), model, None, None, seed=seed, conv_eta=conv_eta,
        learn_conv=True, conv_structure=True, conv_k_max=12, conv_k_min=2,
        conv_grow_mode="split", conv_eta_schedule="cosine", total_steps=total_steps,
        conv_grow_per_burst=0, conv_redundancy_prune=True,
        conv_redundancy_threshold=0.80, conv_redundancy_mode="activation")
    return trainer, model


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=12000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--out", default=_DEFAULT_OUT)
    args = ap.parse_args(argv)

    print(f"loading MNIST-14 (n_train={args.n_train}) ...")
    Xtr, ytr, Xte, yte, scaler = _load_mnist14(args.seed, args.n_train, args.n_test)
    side, n_out = 14, int(max(ytr.max(), yte.max())) + 1

    trainer, model = build_conv_sprout(side, n_out, args.seed, args.steps)
    trainer.X, trainer.y = Xtr.reshape(-1, side, side), ytr
    Xte_img = Xte.reshape(-1, side, side)

    print(f"training conv-sprout for {args.steps} steps ...")
    t0 = time.perf_counter()
    for s in range(args.steps):
        trainer.step()
        if s % 2000 == 0 or s == args.steps - 1:
            acc = trainer.accuracy(Xte_img[:1000], yte[:1000])
            print(f"  step {s:6d}  test_acc(1k)={acc:.4f}  "
                  f"filters={model.conv.n_active}  syn={len(model.head.synapses)}")
    wall = time.perf_counter() - t0

    test_acc = trainer.accuracy(Xte_img, yte)
    meta = {"side": side, "kh": 3, "kw": 3, "pool": 2, "value_scale": 255.0,
            "classes": list(range(n_out)), "test_acc": round(float(test_acc), 4),
            "steps": args.steps, "seed": args.seed,
            "n_active_filters": int(model.conv.n_active),
            "head_synapses": len(model.head.synapses)}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_model(args.out, model, scaler, meta)
    print(f"\nfinal test_acc={test_acc:.4f}  filters={model.conv.n_active}  "
          f"syn={len(model.head.synapses)}  ({wall:.0f}s)")
    print(f"saved -> {args.out}")
    return meta


if __name__ == "__main__":
    main()
