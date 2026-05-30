"""Head-to-head: v1 eligibility baseline vs the gradient-as-currency enhancement.

Both run on the SAME spirals data, network shape, seed and learning rate. We
report the things the enhancement was supposed to improve:

  * accuracy (should match - currency shouldn't cost performance)
  * synapse-count behaviour (grow-then-stabilize, no churn)
  * confidence / effective-LR band (the headline mechanic, should stay legible)
  * structural churn (grow events; baseline wastes growth on dead ReLU units)
  * concept-shift recovery (swap labels mid-life; currency should thaw and
    re-adapt on its own, faster, because contested wires lose confidence)

Writes output/compare/comparison.json and output/compare/compare.png.
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sprout.data import generate_spirals
from sprout.network import build_graph, init_weights
from sprout.train import Config, Trainer, accuracy

OUT = os.path.join("output", "compare")
LAYERS = [2, 10, 10, 8, 2]
DENSITY = 0.4
STEPS = 20000
SHIFT_STEPS = 6000
RECORD_EVERY = 200
SEED = 0


def baseline_cfg():
    # the tuned v1 "full" eligibility system
    return Config(eta_base=0.02, enable_eligibility=True, enable_confidence=True,
                  enable_prune=True, enable_grow=True,
                  theta_prune=0.001, prune_warmup=6000)


def currency_cfg():
    # the enhancement, out of the box: no warmup, no theta, no grow_budget
    return Config(eta_base=0.02, grad_currency=True, enable_confidence=True,
                  enable_prune=True, enable_grow=True,
                  gamma_dec=0.001, t_struct=200)


def currency_tuned_cfg():
    # same, but a longer grace so newborn (weight-0) wires develop before the
    # pruner judges them - the principled replacement for grow_budget, testing
    # whether grace alone tames the grow/prune churn.
    return Config(eta_base=0.02, grad_currency=True, enable_confidence=True,
                  enable_prune=True, enable_grow=True,
                  gamma_dec=0.001, t_struct=200, t_grace=1000, grow_bar_frac=2.0)


def run_one(name, cfg, X, y):
    net = build_graph(list(LAYERS), density=DENSITY, seed=SEED)
    init_weights(net, seed=SEED)
    tr = Trainer(cfg, net, X, y, seed=SEED)
    for s in range(STEPS):
        tr.step(record=(s % RECORD_EVERY == 0 or s == STEPS - 1))

    # concept shift: swap the labels, keep training, watch recovery
    shift_start_step = len(tr.history["rec_step"])
    tr.X, tr.y = X, 1 - y
    for s in range(SHIFT_STEPS):
        tr.step(record=(s % RECORD_EVERY == 0 or s == SHIFT_STEPS - 1))

    h = tr.history
    confs = [s.confidence for s in net.synapses.values()]
    grow_by_target = {}
    for e in tr.events:
        if e["type"] == "grow":
            j = e["edge"][1]
            grow_by_target[j] = grow_by_target.get(j, 0) + 1
    n_prune = sum(1 for e in tr.events if e["type"] == "prune")
    n_grow = sum(1 for e in tr.events if e["type"] == "grow")
    sc = h["synapse_count"]

    # accuracy recovery after the shift (on the new, swapped labels)
    acc_curve = h["accuracy"]
    post = acc_curve[shift_start_step:]
    final_recovered = post[-1] if post else None

    return {
        "name": name,
        "history": h,
        "shift_start_step": shift_start_step,
        "final_accuracy_pre_shift": acc_curve[shift_start_step - 1],
        "max_accuracy_pre_shift": max(acc_curve[:shift_start_step]),
        "recovered_accuracy_post_shift": final_recovered,
        "synapse_count_start": sc[0],
        "synapse_count_peak": max(sc[:STEPS]),
        "synapse_count_end_pre_shift": sc[STEPS - 1],
        "mean_confidence_end": float(np.mean(confs)) if confs else 0.0,
        "max_confidence_end": float(np.max(confs)) if confs else 0.0,
        "mean_eff_lr_start": h["mean_eff_lr"][0],
        "mean_eff_lr_end": h["mean_eff_lr"][shift_start_step - 1],
        "n_prune_events": n_prune,
        "n_grow_events": n_grow,
        "distinct_neurons_grown": len(grow_by_target),
        "max_grows_into_one_neuron": max(grow_by_target.values()) if grow_by_target else 0,
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    X, y = generate_spirals(n=600, seed=SEED, turns=1.0, noise=0.10)

    runs_spec = [
        ("baseline (eligibility)", baseline_cfg(), "tab:blue"),
        ("currency", currency_cfg(), "tab:red"),
        ("currency+grace", currency_tuned_cfg(), "tab:green"),
    ]
    results = []
    for name, cfg, color in runs_spec:
        print(f"running {name} ...")
        r = run_one(name, cfg, X, y)
        r["color"] = color
        results.append(r)

    # ---- table ----
    rows = [
        ("final accuracy (pre-shift)", "final_accuracy_pre_shift", "{:.3f}"),
        ("max accuracy (pre-shift)", "max_accuracy_pre_shift", "{:.3f}"),
        ("recovered acc (post label-swap)", "recovered_accuracy_post_shift", "{:.3f}"),
        ("synapse count start", "synapse_count_start", "{}"),
        ("synapse count peak", "synapse_count_peak", "{}"),
        ("synapse count end (pre-shift)", "synapse_count_end_pre_shift", "{}"),
        ("mean confidence (end)", "mean_confidence_end", "{:.2f}"),
        ("max confidence (end)", "max_confidence_end", "{:.2f}"),
        ("mean eff-LR start", "mean_eff_lr_start", "{:.4f}"),
        ("mean eff-LR end", "mean_eff_lr_end", "{:.4f}"),
        ("prune events", "n_prune_events", "{}"),
        ("grow events", "n_grow_events", "{}"),
        ("distinct neurons grown into", "distinct_neurons_grown", "{}"),
        ("max grows into ONE neuron (churn)", "max_grows_into_one_neuron", "{}"),
    ]
    width = 74 + 18 * (len(results) - 2)
    print("\n" + "=" * width)
    header = f"{'metric':<36}" + "".join(f"{r['name']:>18}" for r in results)
    print(header)
    print("-" * width)
    for label, key, fmt in rows:
        cells = ""
        for r in results:
            v = r[key]
            cells += f"{(fmt.format(v) if v is not None else '-'):>18}"
        print(f"{label:<36}{cells}")
    print("=" * width)

    # ---- plot ----
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for r in results:
        h, color = r["history"], r["color"]
        xs = h["rec_step"]
        ax[0].plot(xs, h["accuracy"], color=color, label=r["name"])
        ax[1].plot(h["step"], h["synapse_count"], color=color, label=r["name"])
        ax[2].plot(xs, h["mean_confidence"], color=color, label=r["name"])
        if r["shift_start_step"] < len(xs):
            ax[0].axvline(xs[r["shift_start_step"]], color=color, ls=":", lw=1, alpha=0.5)
    ax[0].set_title("accuracy (dotted = label swap)"); ax[0].set_xlabel("step"); ax[0].legend()
    ax[1].set_title("synapse count"); ax[1].set_xlabel("step"); ax[1].legend()
    ax[2].set_title("mean confidence"); ax[2].set_xlabel("step"); ax[2].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "compare.png"), dpi=90)
    plt.close(fig)

    # ---- json (drop the bulky history) ----
    def strip(r):
        return {k: v for k, v in r.items() if k not in ("history", "color")}
    with open(os.path.join(OUT, "comparison.json"), "w") as f:
        json.dump({r["name"]: strip(r) for r in results}, f, indent=2, default=float)
    print("artifacts ->", OUT)


if __name__ == "__main__":
    main()
