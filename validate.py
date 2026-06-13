"""Validation harness: check each "it works" criterion with numbers, not vibes.

This validates the CURRENT architecture - gradient-as-currency - in which one
metered signal (the per-synapse backprop gradient) drives confidence, pruning
and growth (see sprout/currency.py).

Produces output/validation/report.json plus supporting plots:
  * eff_lr.png       - mean effective learning rate falling as confidence rises
  * selectivity.png  - the metered signal is selective (demand meter vs fresh gradient)
  * decay.png        - confidence FALLING after a concept shift (re-adaptation)
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

OUT = os.path.join("output", "validation")


def _make_config():
    # validate.py is a FIXED guardrail of the core currency mechanics
    # (confidence/prune/grow) running CONTINUOUSLY. Sleep + phasic structure +
    # startle are pinned OFF here so this 7/7 guardrail stays a stable
    # reference of the base learning loop (those are validated separately
    # under docs/eval-runs).
    return Config(eta_base=0.02, enable_confidence=True, enable_prune=True,
                  enable_grow=True, gamma_dec=0.001, t_struct=200,
                  enable_sleep=False, phasic_structure=False, startle=False)


def main():
    os.makedirs(OUT, exist_ok=True)
    X, y = generate_spirals(n=600, seed=0, turns=1.0, noise=0.10)
    # NOTE: this guardrail is deliberately PINNED to the original (2,10,10,8,2)
    # net @ 30k steps even though the project default moved to w16 @ 15k (see
    # evals/spec.py). Its §11 pass/fail thresholds (confidence gating,
    # consolidation counts, shift decay) were tuned against this exact net and
    # horizon; keeping it fixed makes it a stable regression check rather than a
    # moving target. Re-point + re-tune it only as a deliberate, separate task.
    net = build_graph([2, 10, 10, 8, 2], density=0.4, seed=0)
    init_weights(net, seed=0)
    tr = Trainer(_make_config(), net, X, y, seed=0)
    tr.track(list(net.synapses.keys()))

    STEPS = 30000
    for s in range(STEPS):
        tr.step(record=(s % 200 == 0 or s == STEPS - 1))

    h = tr.history
    report = {"_mode": "currency"}

    # 1. boundary fits spirals -----------------------------------------------
    acc = accuracy(net, X, y)
    report["1_boundary_fits_spirals"] = {
        "final_accuracy": acc, "max_accuracy": max(h["accuracy"]), "pass": acc > 0.9}

    # 2. the metered signal is SELECTIVE -------------------------------------
    report["2_signal_selective"] = _currency_meter_fidelity(net, X, y)

    # 3. confidence gates learning on the wires it CONSOLIDATES --------------
    # 2D currency confidence is selective by design: it freezes only the few
    # load-bearing wires (important AND settled) and leaves the majority plastic,
    # so the POPULATION-MEAN effective LR barely moves (the old mean-drop check
    # was calibrated for the more uniform legacy confidence and understates the
    # 2D rule). The faithful check is that the consolidated wires (top-decile
    # confidence) get their learning gated down, and that meaningful consolidation
    # happens at all (>=1 wire with its rate at least halved). Legacy confidence is
    # more uniform and clears this comfortably too. (mean_eff_lr_* kept for the
    # eff_lr.png trend.)
    lr0, lr1 = h["mean_eff_lr"][0], h["mean_eff_lr"][-1]
    confs = np.array([s.confidence for s in net.synapses.values()])
    eff_lr = tr.cfg.eta_base / (1.0 + confs)
    top = confs >= np.quantile(confs, 0.9)            # the consolidated decile
    consolidated_eff_lr = float(eff_lr[top].mean())
    consolidated_gate_x = (tr.cfg.eta_base / consolidated_eff_lr
                           if consolidated_eff_lr else None)
    report["3_confidence_gates_learning"] = {
        "mean_eff_lr_start": lr0, "mean_eff_lr_end": lr1,
        "mean_confidence_end": float(confs.mean()), "max_confidence_end": float(confs.max()),
        "consolidated_eff_lr": consolidated_eff_lr,
        "consolidated_gate_x": consolidated_gate_x,
        "pass": (consolidated_gate_x or 0) > 1.5 and float(confs.max()) > 1.0}
    _plot_eff_lr(h)

    # 5. low-utility pruned without tanking accuracy --------------------------
    n_prune = sum(1 for e in tr.events if e["type"] == "prune")
    report["5_pruning_without_collapse"] = {
        "n_prune_events": n_prune, "final_accuracy": acc,
        "pass": n_prune > 5 and acc > 0.9}

    # 6. growth adds useful wires; newborns start at weight 0 ----------------
    n_grow = sum(1 for e in tr.events if e["type"] == "grow")
    born_zero, dead_skipped = _check_growth()
    # the RigL win: growth never targets a dead (zero-gradient) neuron
    crit6 = {"n_grow_events": n_grow, "newborns_start_at_zero_weight": born_zero,
             "dead_neuron_never_grown": dead_skipped,
             "pass": n_grow > 5 and born_zero and dead_skipped}
    report["6_growth_useful"] = crit6

    # 7. synapse count grows then stabilizes ---------------------------------
    sc = np.array(h["synapse_count"])
    peak_i = int(np.argmax(sc))
    grew_early = sc[peak_i] > sc[0] + 5 and peak_i < len(sc) // 2
    tail = sc[-len(sc) // 5:]
    stabilized = (tail.max() - tail.min()) <= 0.1 * sc[0] + 3
    report["7_count_grows_then_stabilizes"] = {
        "start": int(sc[0]), "peak": int(sc[peak_i]), "end": int(sc[-1]),
        "tail_range": int(tail.max() - tail.min()),
        "grew_early": bool(grew_early), "stabilized": bool(stabilized),
        "pass": bool(grew_early and stabilized)}

    # 4. confidence can FALL (decay) — concept-shift demonstration ------------
    report["4_confidence_can_decay"] = _decay_demo(tr, net, X, y)

    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)

    print("\n============ VALIDATION (currency) ============")
    for k in sorted(report):
        if k.startswith("_"):
            continue
        r = report[k]
        print(("[PASS] " if r.get("pass") else "[FAIL] ") + k)
        for kk, vv in r.items():
            if kk != "pass":
                print(f"         {kk}: {vv}")
    n_pass = sum(1 for k, r in report.items()
                 if not k.startswith("_") and r.get("pass"))
    n_crit = sum(1 for k in report if not k.startswith("_"))
    print(f"\n  {n_pass}/{n_crit} criteria pass")
    print("  artifacts ->", OUT)
    return report


# -- criterion 2 variants ----------------------------------------------------

def _currency_meter_fidelity(net, X, y):
    """The currency primitive is faithful: each wire's metered ``grad_mag`` (an
    EMA of |dL/dw|) tracks the gradient the loss actually applies, measured
    fresh over the whole dataset. High correlation => the 'currency' the three
    readouts spend is a real, selective signal, not noise."""
    fresh = {k: 0.0 for k in net.synapses}
    for xi, yi in zip(X, y):
        net.forward(xi)
        _, gw, _ = net.backward(int(yi))
        for k in net.synapses:
            fresh[k] += abs(gw[k])
    n = len(X)
    fresh = {k: v / n for k, v in fresh.items()}
    mk = np.array([net.synapses[k].grad_mag for k in net.synapses])
    fk = np.array([fresh[k] for k in net.synapses])
    corr = float(np.corrcoef(mk, fk)[0, 1])
    # most- vs least-demanded third (the meter separates settled from pushed)
    order = np.argsort(fk)
    lo = mk[order[: len(order) // 3]].mean()
    hi = mk[order[-len(order) // 3:]].mean()
    _plot_selectivity(fk, mk, "fresh mean |dL/dw|", "metered grad_mag",
                      "Demand meter tracks the true gradient")
    return {"corr_meter_vs_fresh_grad": corr,
            "mean_meter_low_demand": float(lo), "mean_meter_high_demand": float(hi),
            "pass": corr > 0.5 and hi > 3 * (lo + 1e-9)}


# -- criterion 6 variants ----------------------------------------------------

def _check_growth():
    """Return ``(newborns_born_at_zero, dead_neuron_skipped)``.

    Grown synapses are born at weight 0 (a no-op on arrival). The currency grower
    additionally must NOT grow into a dead neuron (one whose gradient is
    identically zero), which is the whole point of RigL-style growth.
    """
    from sprout.currency import batch_edge_scores, grow_currency
    from sprout.data import generate_blobs
    net = build_graph([2, 4, 4, 2], density=0.5, seed=1)
    init_weights(net, seed=1)
    dead = net.layers[2][0]
    net.neurons[dead].bias = -1e6                # permanently silent (zero grad)
    X, y = generate_blobs(n=64, seed=0)
    ghost, ref = batch_edge_scores(net, X, y)
    before = set(net.synapses)
    grow_currency(net, ghost, ref, max_grow=3, grow_bar_frac=0.0)  # grow freely
    new = set(net.synapses) - before
    born_zero = len(new) >= 1 and all(net.synapses[k].weight == 0.0 for k in new)
    dead_skipped = all(j != dead for (_, j) in new)
    return born_zero, dead_skipped


# -- shared: concept-shift decay --------------------------------------------

def _decay_demo(tr, net, X, y):
    """After consolidation, swap the labels (concept shift). Previously-useful
    synapses stop matching the data, so their confidence is pulled DOWN: in
    legacy via loss-gated decay; in currency because the once-settled wires
    suddenly become contested hot-spots (d>1, kappa drops). We track the mean
    confidence of the synapses that were most confident before the shift."""
    survivors = [(k, traj) for k, traj in tr.tracked.items()
                 if k in net.synapses and traj and traj[-1] is not None]
    survivors.sort(key=lambda kt: -(kt[1][-1] or 0))
    top = [k for k, _ in survivors[:10]]
    pre_shift_conf = float(np.mean([net.synapses[k].confidence for k in top]))

    tr.X, tr.y = X, 1 - y
    rel = {k: [] for k in top}
    SHIFT_STEPS = 8000
    for s in range(SHIFT_STEPS):
        tr.step(record=False)
        if s % 100 == 0:
            for k in top:
                syn = net.synapses.get(k)
                rel[k].append(syn.confidence if syn else None)

    mins = []
    for k in top:
        vals = [v for v in rel[k] if v is not None]
        if vals:
            mins.append(min(vals))
    min_after = float(np.mean(mins)) if mins else pre_shift_conf

    _plot_decay(top, rel, pre_shift_conf)
    return {"mean_conf_top10_before_shift": pre_shift_conf,
            "mean_min_conf_after_shift": min_after,
            "fell_by": pre_shift_conf - min_after,
            "pass": min_after < 0.8 * pre_shift_conf}


# -- plots -------------------------------------------------------------------

def _plot_eff_lr(h):
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(h["rec_step"], h["mean_eff_lr"], color="tab:blue", label="mean effective LR")
    ax1.set_xlabel("step"); ax1.set_ylabel("mean η/(1+c)", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(h["rec_step"], h["mean_confidence"], color="tab:red", label="mean confidence")
    ax2.set_ylabel("mean confidence", color="tab:red")
    ax1.set_title("Confidence rises → effective learning rate falls")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "eff_lr.png"), dpi=90); plt.close(fig)


def _plot_selectivity(xv, yv, xlabel, ylabel, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(xv, yv, s=12, alpha=0.6)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "selectivity.png"), dpi=90); plt.close(fig)


def _plot_decay(top, rel, pre):
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = [i * 100 for i in range(len(next(iter(rel.values()))))]
    for k in top:
        vals = rel[k]
        ax.plot(xs, [v if v is not None else np.nan for v in vals], alpha=0.6)
    ax.axhline(pre, color="k", ls="--", lw=1, label=f"mean conf before shift = {pre:.2f}")
    ax.set_xlabel("steps after label-swap (concept shift)")
    ax.set_ylabel("confidence of previously-confident synapses")
    ax.set_title("Confidence DECAYS when synapses stop being useful")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "decay.png"), dpi=90); plt.close(fig)


if __name__ == "__main__":
    main()
