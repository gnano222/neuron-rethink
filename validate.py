"""Validation harness (§11): check each "it works" criterion with numbers,
not vibes. Produces output/validation/report.json plus supporting plots:

  * eff_lr.png       - mean effective learning rate falling as confidence rises
  * eligibility.png  - eligibility vs co-activation (gate is selective)
  * decay.png        - confidence FALLING after a concept shift (the §11
                       "confidence can decay" criterion, shown on a live run)
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


def main():
    os.makedirs(OUT, exist_ok=True)
    X, y = generate_spirals(n=600, seed=0, turns=1.0, noise=0.10)
    net = build_graph([2, 10, 10, 8, 2], density=0.4, seed=0)
    init_weights(net, seed=0)
    cfg = Config(eta_base=0.02, enable_eligibility=True, enable_confidence=True,
                 enable_prune=True, enable_grow=True, theta_prune=0.001, prune_warmup=6000)
    tr = Trainer(cfg, net, X, y, seed=0)
    tr.track(list(net.synapses.keys()))  # track every initial synapse's confidence

    STEPS = 30000
    for s in range(STEPS):
        tr.step(record=(s % 200 == 0 or s == STEPS - 1))

    h = tr.history
    report = {}

    # 1. boundary fits spirals -----------------------------------------------
    acc = accuracy(net, X, y)
    report["1_boundary_fits_spirals"] = {
        "final_accuracy": acc, "max_accuracy": max(h["accuracy"]), "pass": acc > 0.9}

    # 2. eligibility high on co-active synapses, low elsewhere ----------------
    coact = {k: 0.0 for k in net.synapses}
    nb = 0
    for xi in X:
        net.forward(xi)
        for (pre, post) in net.synapses:
            coact[(pre, post)] += abs(net.neurons[pre].activation * net.neurons[post].activation)
        nb += 1
    coact = {k: v / nb for k, v in coact.items()}
    elig = {k: net.synapses[k].eligibility for k in net.synapses}
    ck = np.array([coact[k] for k in net.synapses])
    ek = np.array([elig[k] for k in net.synapses])
    corr = float(np.corrcoef(ck, ek)[0, 1])
    # eligibility of the most vs least co-active third
    order = np.argsort(ck)
    lo = ek[order[: len(order) // 3]].mean()
    hi = ek[order[-len(order) // 3:]].mean()
    report["2_eligibility_tracks_coactivation"] = {
        "corr_elig_vs_coact": corr, "mean_elig_low_coact": float(lo),
        "mean_elig_high_coact": float(hi), "pass": corr > 0.5 and hi > 5 * (lo + 1e-9)}

    # 3. confidence rises on useful synapses & effective LR drops -------------
    lr0, lr1 = h["mean_eff_lr"][0], h["mean_eff_lr"][-1]
    confs = np.array([s.confidence for s in net.synapses.values()])
    report["3_confidence_gates_learning"] = {
        "mean_eff_lr_start": lr0, "mean_eff_lr_end": lr1,
        "lr_dropped_x": lr0 / lr1 if lr1 else None,
        "mean_confidence_end": float(confs.mean()), "max_confidence_end": float(confs.max()),
        "pass": lr1 < 0.7 * lr0 and confs.mean() > 0.5}
    _plot_eff_lr(h)
    _plot_eligibility(ck, ek)

    # 5. low-utility pruned without tanking accuracy --------------------------
    n_prune = sum(1 for e in tr.events if e["type"] == "prune")
    report["5_pruning_without_collapse"] = {
        "n_prune_events": n_prune, "final_accuracy": acc,
        "pass": n_prune > 5 and acc > 0.9}

    # 6. underfiring neurons grow; newborns start at weight 0 ----------------
    n_grow = sum(1 for e in tr.events if e["type"] == "grow")
    # the grow() code constructs newborns at weight 0 by design; assert no grow
    # event coincided with a non-zero birth weight via a fresh probe:
    born_zero = _check_newborn_zero_weight()
    report["6_growth_into_underfiring"] = {
        "n_grow_events": n_grow, "newborns_start_at_zero_weight": born_zero,
        "pass": n_grow > 5 and born_zero}

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

    print("\n================ §11 VALIDATION =================")
    for k in sorted(report):
        r = report[k]
        print(("[PASS] " if r.get("pass") else "[FAIL] ") + k)
        for kk, vv in r.items():
            if kk != "pass":
                print(f"         {kk}: {vv}")
    n_pass = sum(1 for r in report.values() if r.get("pass"))
    print(f"\n  {n_pass}/{len(report)} criteria pass")
    print("  artifacts ->", OUT)
    return report


def _check_newborn_zero_weight():
    """Force one growth event on a net with a free candidate and confirm the
    newborn synapse is born at weight 0 (no disruption)."""
    from sprout.network import Network
    from sprout.plasticity import grow
    net = Network([3, 3, 2])              # layer0={0,1,2}, layer1={3,4,5}
    for n in net.neurons:
        n.firing_rate = 1.0
    net.add_synapse(0, 3)                 # neuron 3 has only one input -> candidates 1,2 free
    net.neurons[3].firing_rate = 0.0      # underfiring
    net.neurons[1].activation = 0.9       # strongest candidate
    before = set(net.synapses)
    grow(net, r_target=0.15, f_under=0.5, max_grow=1, grow_budget=6)
    new = set(net.synapses) - before
    return len(new) >= 1 and all(net.synapses[k].weight == 0.0 for k in new)


def _decay_demo(tr, net, X, y):
    """After consolidation, swap the labels (concept shift). Previously-useful
    synapses stop matching the data -> loss rises -> g->0 -> confidence credit
    stops -> slow decay pulls confidence DOWN. We track the mean confidence of
    the synapses that were most confident before the shift."""
    # most-confident surviving tracked synapses (full trajectory available)
    survivors = [(k, traj) for k, traj in tr.tracked.items()
                 if k in net.synapses and traj and traj[-1] is not None]
    survivors.sort(key=lambda kt: -(kt[1][-1] or 0))
    top = [k for k, _ in survivors[:10]]
    pre_shift_conf = float(np.mean([net.synapses[k].confidence for k in top]))

    y_swapped = 1 - y
    tr.X, tr.y = X, y_swapped
    rel = {k: [] for k in top}
    SHIFT_STEPS = 8000
    for s in range(SHIFT_STEPS):
        tr.step(record=False)
        if s % 100 == 0:
            for k in top:
                syn = net.synapses.get(k)
                rel[k].append(syn.confidence if syn else None)

    # minimum confidence reached during the shift (the "fall")
    mins = []
    for k in top:
        vals = [v for v in rel[k] if v is not None]
        if vals:
            mins.append(min(vals))
    min_after = float(np.mean(mins)) if mins else pre_shift_conf

    _plot_decay(tr, top, rel, pre_shift_conf)
    return {"mean_conf_top10_before_shift": pre_shift_conf,
            "mean_min_conf_after_shift": min_after,
            "fell_by": pre_shift_conf - min_after,
            "pass": min_after < 0.8 * pre_shift_conf}


def _plot_eff_lr(h):
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(h["rec_step"], h["mean_eff_lr"], color="tab:blue", label="mean effective LR")
    ax1.set_xlabel("step"); ax1.set_ylabel("mean η/(1+c)", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(h["rec_step"], h["mean_confidence"], color="tab:red", label="mean confidence")
    ax2.set_ylabel("mean confidence", color="tab:red")
    ax1.set_title("Confidence rises → effective learning rate falls")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "eff_lr.png"), dpi=90); plt.close(fig)


def _plot_eligibility(coact, elig):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(coact, elig, s=12, alpha=0.6)
    ax.set_xlabel("mean co-activation |a_pre · a_post|")
    ax.set_ylabel("eligibility")
    ax.set_title("Eligibility is high only on co-active synapses")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "eligibility.png"), dpi=90); plt.close(fig)


def _plot_decay(tr, top, rel, pre):
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
