"""Conv-SPROUT (Phase 2) experiment runner.

Trains the weight-shared conv economy + sparse phasic head jointly across seeds and
arms, then writes a committed, phone-readable run folder (metrics table with a
seed-bootstrap verdict, accuracy curves, and a FILTER VISUALIZATION for the
legibility claim). Arms differ only in the conv front-end (fixed vs learned vs
self-sizing); the head is the same sparse phasic-startle-k4-style economy.

    python conv_experiment.py --arms fixed-hand-k6,learned-k6,learned-k12 \
        --seeds 5 --steps 15000 --name conv-sprout-e1-learned-vs-fixed
"""
from __future__ import annotations

import argparse
import json
import os
import time

# keep BLAS single-threaded per worker (matches evals/runner.py)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from concurrent.futures import ProcessPoolExecutor   # noqa: E402

import numpy as np                                    # noqa: E402

from sprout.conv import ConvEconomy                   # noqa: E402
from sprout.conv_features import filter_bank          # noqa: E402
from sprout.conv_train import ConvModel, ConvTrainer  # noqa: E402
from sprout.datasets import get_dataset               # noqa: E402
from sprout.network import build_graph, init_weights  # noqa: E402
from sprout.train import Config                        # noqa: E402


# arm name -> conv front-end spec (head is identical across arms).
# eta_sched controls filter-LR CONSOLIDATION: "none" | "cosine" | "freeze".
ARMS = {
    "fixed-hand-k6":  dict(k_max=6,  k_init=6,  init="hand",   learn=True,  structure=False, freeze=True,  grow_mode="split"),
    # consolidation arms: learn filters early, wind LR down so they settle late
    "learned-k6-cos": dict(k_max=6,  k_init=6,  init="random", learn=True,  structure=False, freeze=False, grow_mode="split", eta_sched="cosine"),
    "selfsize-2to12-rand-cos": dict(k_max=12, k_init=2, init="random", learn=True, structure=True, freeze=False, grow_mode="random", eta_sched="cosine"),
    # redundancy pruning: start FULL (12), no growth, trim duplicate filters
    # (high kernel cosine similarity) so the bank leans out to its natural size.
    "selfsize-12to12-cos":         dict(k_max=12, k_init=12, init="random", learn=True, structure=True, freeze=False, grow_mode="split", eta_sched="cosine", grow_per_burst=0, redprune=False),
    "selfsize-12to12-redprune-cos": dict(k_max=12, k_init=12, init="random", learn=True, structure=True, freeze=False, grow_mode="split", eta_sched="cosine", grow_per_burst=0, redprune=True, red_thresh=0.85),
    # FUNCTIONAL redundancy: prune filters whose OUTPUTS are correlated (carry the
    # same info), not just whose kernels match -- the stronger leaning-out signal.
    "selfsize-12to12-actprune-cos": dict(k_max=12, k_init=12, init="random", learn=True, structure=True, freeze=False, grow_mode="split", eta_sched="cosine", grow_per_burst=0, redprune=True, red_mode="activation", red_thresh=0.9),
    # BOTH signals: prune if kernel-similar (0.85) OR output-correlated (0.9)
    "selfsize-12to12-bothprune-cos": dict(k_max=12, k_init=12, init="random", learn=True, structure=True, freeze=False, grow_mode="split", eta_sched="cosine", grow_per_burst=0, redprune=True, red_mode="both", red_thresh=0.9),
    # FUNCTIONAL-prune CUTOFF SWEEP: same arm, only the activation-correlation
    # threshold varies (lower = prune more aggressively). Maps trim vs accuracy.
    "actsweep-t90": dict(k_max=12, k_init=12, init="random", learn=True, structure=True, freeze=False, grow_mode="split", eta_sched="cosine", grow_per_burst=0, redprune=True, red_mode="activation", red_thresh=0.90),
    "actsweep-t80": dict(k_max=12, k_init=12, init="random", learn=True, structure=True, freeze=False, grow_mode="split", eta_sched="cosine", grow_per_burst=0, redprune=True, red_mode="activation", red_thresh=0.80),
    "actsweep-t70": dict(k_max=12, k_init=12, init="random", learn=True, structure=True, freeze=False, grow_mode="split", eta_sched="cosine", grow_per_burst=0, redprune=True, red_mode="activation", red_thresh=0.70),
    "actsweep-t60": dict(k_max=12, k_init=12, init="random", learn=True, structure=True, freeze=False, grow_mode="split", eta_sched="cosine", grow_per_burst=0, redprune=True, red_mode="activation", red_thresh=0.60),
    "actsweep-t50": dict(k_max=12, k_init=12, init="random", learn=True, structure=True, freeze=False, grow_mode="split", eta_sched="cosine", grow_per_burst=0, redprune=True, red_mode="activation", red_thresh=0.50),
    # tight-budget arms: under scarcity, WHICH filters you keep should matter
    "fixed-hand-k2":  dict(k_max=2,  k_init=2,  init="hand",   learn=True,  structure=False, freeze=True,  grow_mode="split"),
    "learned-k2":     dict(k_max=2,  k_init=2,  init="random", learn=True,  structure=False, freeze=False, grow_mode="split"),
    "fixed-hand-k3":  dict(k_max=3,  k_init=3,  init="hand",   learn=True,  structure=False, freeze=True,  grow_mode="split"),
    "learned-k3":     dict(k_max=3,  k_init=3,  init="random", learn=True,  structure=False, freeze=False, grow_mode="split"),
    "learned-k6":     dict(k_max=6,  k_init=6,  init="random", learn=True,  structure=False, freeze=False, grow_mode="split"),
    "learned-k12":    dict(k_max=12, k_init=12, init="random", learn=True,  structure=False, freeze=False, grow_mode="split"),
    # E2 self-sizing: start few, let the economy grow/prune filters up to k_max
    "selfsize-2to12-split": dict(k_max=12, k_init=2, init="random", learn=True, structure=True, freeze=False, grow_mode="split"),
    "selfsize-2to12-rand":  dict(k_max=12, k_init=2, init="random", learn=True, structure=True, freeze=False, grow_mode="random"),
    # start ABOVE the useful count to test whether the economy PRUNES redundancy
    "selfsize-12to12-prune": dict(k_max=12, k_init=12, init="random", learn=True, structure=True, freeze=False, grow_mode="split"),
}


def _motif_images(seed, n_points, n_test=1000, side=12, n_classes=6, n_stamp=2):
    """Positive-control task where FILTER QUALITY is the bottleneck: each class is
    a fixed random 3x3 motif stamped at random positions (translation-invariant).
    The optimal detectors are MATCHED filters for the motifs -- which a generic
    edge/blob bank does not contain -- so learned filters should beat fixed here
    even though they tie on MNIST. Motifs are task-fixed (seed 12345); the data
    seed only varies images/positions. Standardized per-pixel on train stats."""
    motifs = np.random.default_rng(12345).normal(size=(n_classes, 3, 3))
    motifs /= np.linalg.norm(motifs.reshape(n_classes, -1), axis=1)[:, None, None]

    def gen(rng, n):
        X = np.zeros((n, side, side))
        y = np.zeros(n, int)
        for i in range(n):
            c = i % n_classes
            for _ in range(n_stamp):
                r, cc = int(rng.integers(0, side - 2)), int(rng.integers(0, side - 2))
                X[i, r:r + 3, cc:cc + 3] += 2.0 * motifs[c]
            y[i] = c
        return X + 0.3 * rng.normal(size=X.shape), y

    Xtr, ytr = gen(np.random.default_rng(seed), n_points)
    Xte, yte = gen(np.random.default_rng(seed + 10000), n_test)
    flat_tr, flat_te = Xtr.reshape(len(Xtr), -1), Xte.reshape(len(Xte), -1)
    mu, sd = flat_tr.mean(0), flat_tr.std(0) + 1e-8
    Xtr = ((flat_tr - mu) / sd).reshape(-1, side, side)
    Xte = ((flat_te - mu) / sd).reshape(-1, side, side)
    return Xtr, ytr, Xte, yte, side


def _images(dataset, seed, n_points):
    if dataset == "motifs":
        return _motif_images(seed, n_points)
    Xtr, ytr, Xte, yte = get_dataset(dataset, seed, n_points=n_points)
    side = int(round(np.sqrt(Xtr.shape[1])))
    return (Xtr.reshape(-1, side, side), ytr, Xte.reshape(-1, side, side), yte, side)


def _head_config():
    """The sparse phasic head economy (w32-sparse / phasic-startle-k4 settings)."""
    return Config(eta_base=0.02, enable_confidence=True, enable_prune=True,
                  enable_grow=True, gamma_dec=0.001, t_struct=200,
                  phasic_structure=True, startle=False, grow_demand_k=4,
                  sleep_warmup=2500, sleep_patience=1500)


def _build(arm, seed, side, conv_eta, n_out=10, total_steps=None):
    spec = ARMS[arm]
    kh = kw = 3
    if spec["init"] == "hand":
        conv = ConvEconomy(k_max=spec["k_max"], kh=kh, kw=kw,
                           kernels=filter_bank("hand")[:spec["k_max"]], seed=seed)
    else:
        conv = ConvEconomy(k_max=spec["k_max"], kh=kh, kw=kw,
                           k_init=spec["k_init"], seed=seed)
    feat = conv.feat_dim(side, side)
    head = build_graph([feat, 32, n_out], density=0.5, seed=seed)
    init_weights(head, seed=seed)
    model = ConvModel(conv, head, side, side)
    cfg = _head_config()
    tr = ConvTrainer(cfg, model, None, None, seed=seed, conv_eta=conv_eta,
                     learn_conv=not spec["freeze"], conv_structure=spec["structure"],
                     conv_k_max=spec["k_max"], conv_k_min=2,
                     conv_grow_mode=spec["grow_mode"],
                     conv_eta_schedule=spec.get("eta_sched", "none"),
                     total_steps=total_steps,
                     conv_grow_per_burst=spec.get("grow_per_burst", 2),
                     conv_redundancy_prune=spec.get("redprune", False),
                     conv_redundancy_threshold=spec.get("red_thresh", 0.9),
                     conv_redundancy_mode=spec.get("red_mode", "kernel"))
    return tr, model


def run_one(arm, seed, args):
    Xtr, ytr, Xte, yte, side = _images(args.dataset, seed, args.points)
    n_out = int(max(ytr.max(), yte.max())) + 1
    tr, model = _build(arm, seed, side, args.conv_eta, n_out=n_out,
                       total_steps=args.steps)
    tr.X, tr.y = Xtr, ytr
    cap = min(args.train_eval_cap, len(Xtr))
    eval_idx = np.sort(np.random.default_rng(seed + 7).choice(len(Xtr), cap, replace=False))
    Xtr_eval, ytr_eval = Xtr[eval_idx], ytr[eval_idx]

    rec_step, test_acc, train_acc, n_active, head_syn = [], [], [], [], []
    t0 = time.perf_counter()
    for s in range(args.steps):
        tr.step()
        if s % args.record_every == 0 or s == args.steps - 1:
            rec_step.append(s)
            test_acc.append(tr.accuracy(Xte, yte))
            train_acc.append(tr.accuracy(Xtr_eval, ytr_eval))
            n_active.append(model.conv.n_active)
            head_syn.append(len(model.head.synapses))
    wall = time.perf_counter() - t0

    ev = tr.events
    return {
        "arm": arm, "seed": seed,
        "rec_step": rec_step, "test_acc": test_acc, "train_acc": train_acc,
        "n_active": n_active, "head_syn": head_syn,
        "final_test_acc": test_acc[-1], "max_test_acc": float(np.max(test_acc)),
        "n_active_end": model.conv.n_active,
        "head_syn_end": len(model.head.synapses),
        "n_conv_grow": sum(1 for e in ev if e["type"] == "conv_grow"),
        "n_conv_prune": sum(1 for e in ev if e["type"] == "conv_prune"),
        "n_sleep": sum(1 for e in ev if e["type"] == "sleep"),
        "wall_time": wall,
        "filters": model.conv.theta[model.conv.active].tolist(),
    }


def _job(t):
    return run_one(*t)


def _verdict(base, arm, n_boot=10000, seed=0):
    base, arm = np.array(base), np.array(arm)
    rng = np.random.default_rng(seed)
    n = len(base)
    diffs = [np.mean(arm[idx] - base[idx])
             for idx in (rng.integers(0, n, n) for _ in range(n_boot))]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return "UP" if lo > 0 else ("DOWN" if hi < 0 else "~")


def _agg(results, arms, baseline):
    by = {a: [r for r in results if r["arm"] == a] for a in arms}
    base_final = [r["final_test_acc"] for r in by[baseline]]
    rows = {}
    for a in arms:
        rs = by[a]
        finals = [r["final_test_acc"] for r in rs]
        maxes = [r["max_test_acc"] for r in rs]
        rows[a] = {
            "final_mean": float(np.mean(finals)), "final_std": float(np.std(finals)),
            "max_mean": float(np.mean(maxes)), "max_std": float(np.std(maxes)),
            "n_active_end": float(np.mean([r["n_active_end"] for r in rs])),
            "head_syn_end": float(np.mean([r["head_syn_end"] for r in rs])),
            "conv_grow": float(np.mean([r["n_conv_grow"] for r in rs])),
            "conv_prune": float(np.mean([r["n_conv_prune"] for r in rs])),
            "wall": float(np.mean([r["wall_time"] for r in rs])),
            "verdict": "(baseline)" if a == baseline else _verdict(base_final, finals),
        }
    return rows, by


def _write(results, args, dest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(dest, exist_ok=True)
    arms = args.arms
    rows, by = _agg(results, arms, args.baseline)

    # accuracy curves (mean over seeds)
    plt.figure(figsize=(7, 4.5))
    for a in arms:
        rs = by[a]
        steps = rs[0]["rec_step"]
        m = np.mean([r["test_acc"] for r in rs], axis=0)
        plt.plot(steps, m, label=a)
    plt.xlabel("step"); plt.ylabel("test accuracy"); plt.legend(); plt.grid(alpha=.3)
    plt.title(args.name); plt.tight_layout()
    plt.savefig(os.path.join(dest, "acc_curves.png"), dpi=110); plt.close()

    # active-filter-count trajectory (the self-sizing signal)
    plt.figure(figsize=(7, 4.5))
    for a in arms:
        rs = by[a]
        steps = rs[0]["rec_step"]
        m = np.mean([r["n_active"] for r in rs], axis=0)
        plt.plot(steps, m, label=a)
    plt.xlabel("step"); plt.ylabel("active filters"); plt.legend(); plt.grid(alpha=.3)
    plt.title(f"{args.name} — filter count"); plt.tight_layout()
    plt.savefig(os.path.join(dest, "count_curves.png"), dpi=110); plt.close()

    # filter visualization for each learned arm (seed-0 final filters)
    for a in arms:
        r0 = next((r for r in by[a] if r["seed"] == 0), by[a][0])
        filt = np.array(r0["filters"])
        if len(filt) == 0:
            continue
        n = len(filt)
        cols = min(6, n); rows_ = int(np.ceil(n / cols))
        plt.figure(figsize=(cols * 1.2, rows_ * 1.2))
        vmax = np.abs(filt).max() + 1e-9
        for i in range(n):
            plt.subplot(rows_, cols, i + 1)
            plt.imshow(filt[i], cmap="RdBu", vmin=-vmax, vmax=vmax)
            plt.axis("off")
        plt.suptitle(f"{a} filters (seed 0)"); plt.tight_layout()
        plt.savefig(os.path.join(dest, f"filters_{a}.png"), dpi=110); plt.close()

    with open(os.path.join(dest, "metrics.json"), "w") as f:
        json.dump({"args": vars(args), "rows": rows, "results": results}, f)

    lines = [f"# Conv-SPROUT Phase 2 — {args.name}", "",
             f"- **Dataset:** {args.dataset}  |  **Seeds:** {args.seeds}  "
             f"|  **Steps:** {args.steps}  |  **Baseline:** {args.baseline}",
             f"- **Head:** sparse phasic (w32-sparse economy), conv 3x3 + ReLU + 2x2 maxpool",
             "", "## Results (mean ± std across seeds)", "",
             "| Arm | final test acc | max test acc | filters end | head synapses | conv grow/prune | verdict vs base |",
             "|---|---|---|---|---|---|---|"]
    for a in arms:
        rw = rows[a]
        lines.append(
            f"| {a} | {rw['final_mean']:.3f} ± {rw['final_std']:.3f} | "
            f"{rw['max_mean']:.3f} ± {rw['max_std']:.3f} | {rw['n_active_end']:.1f} | "
            f"{rw['head_syn_end']:.0f} | {rw['conv_grow']:.1f}/{rw['conv_prune']:.1f} | "
            f"{rw['verdict']} |")
    lines += ["", "Verdict = 95% seed-bootstrap CI of the final-test-acc difference "
              "vs the baseline (UP/DOWN/~).", "",
              "![acc](acc_curves.png)", "", "![filter count](count_curves.png)", ""]
    for a in arms:
        if os.path.exists(os.path.join(dest, f"filters_{a}.png")):
            lines.append(f"### {a} learned filters\n\n![{a}](filters_{a}.png)\n")
    with open(os.path.join(dest, "README.md"), "w") as f:
        f.write("\n".join(lines))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="fixed-hand-k6,learned-k6,learned-k12")
    ap.add_argument("--baseline", default="fixed-hand-k6")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--points", type=int, default=12000)
    ap.add_argument("--train-eval-cap", type=int, default=2000)
    ap.add_argument("--record-every", type=int, default=1000)
    ap.add_argument("--conv-eta", type=float, default=0.02)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--name", default="conv-sprout-run")
    ap.add_argument("--out-root", default="docs/eval-runs")
    args = ap.parse_args(argv)
    args.arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    assert args.baseline in args.arms

    tasks = [(a, s, args) for a in args.arms for s in range(args.seeds)]
    print(f"running {len(tasks)} jobs ({len(args.arms)} arms x {args.seeds} seeds, "
          f"{args.steps} steps) ...")
    if args.jobs == 1:
        results = [_job(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            results = list(ex.map(_job, tasks))

    dest = os.path.join(args.out_root, args.name)
    rows = _write(results, args, dest)
    print(f"\npublished -> {dest}")
    for a in args.arms:
        rw = rows[a]
        print(f"  {a:18s} final {rw['final_mean']:.3f}±{rw['final_std']:.3f}  "
              f"max {rw['max_mean']:.3f}  filters {rw['n_active_end']:.1f}  "
              f"{rw['verdict']}")
    return rows


if __name__ == "__main__":
    main()
