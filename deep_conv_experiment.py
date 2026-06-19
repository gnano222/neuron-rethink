"""Deep (stacked) Conv-SPROUT experiment runner.

Compares conv depth-1 vs depth-2 (compositional features) on full-res 28x28 MNIST,
where the README's depth study says depth has the spatial headroom 14x14 lacks.
Both arms share the SAME first conv layer + the SAME phasic-startle-k4 head and
cosine filter consolidation; depth-2 just adds a second multi-channel conv layer,
so DEPTH is the only variable. Writes a committed, phone-readable run folder.

    python deep_conv_experiment.py --arms depth1,depth2 --baseline depth1 \
        --seeds 5 --steps 30000 --name deep-conv-28
"""
from __future__ import annotations

import argparse
import json
import os
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from concurrent.futures import ProcessPoolExecutor   # noqa: E402

import numpy as np                                    # noqa: E402

from sprout.conv_deep import ConvLayer, DeepConvModel, DeepConvTrainer  # noqa: E402
from sprout.datasets import get_dataset               # noqa: E402
from sprout.network import build_graph, init_weights  # noqa: E402
from sprout.train import Config                        # noqa: E402


# arm -> conv channel widths per layer (c_in chains automatically). depth-2 shares
# depth-1's first layer (k=8) and adds a 16-filter multi-channel layer on top.
ARMS = {
    "depth1": dict(channels=[8]),
    "depth2": dict(channels=[8, 16]),
    "depth1-wide": dict(channels=[16]),   # capacity control: more L1 filters, no depth
}


def _head_config():
    """The sparse phasic head economy (matches conv_experiment._head_config)."""
    return Config(eta_base=0.02, enable_confidence=True, enable_prune=True,
                  enable_grow=True, gamma_dec=0.001, t_struct=200,
                  phasic_structure=True, startle=False, grow_demand_k=4,
                  sleep_warmup=2500, sleep_patience=1500)


def _feat_dim(layers, h, w):
    ch, cw = h, w
    for layer in layers:
        ch, cw = layer.out_hw(ch, cw)
    return layers[-1].k * ch * cw


def _conv_params(layers):
    return int(sum(layer.theta.size for layer in layers))


def _build(arm, seed, side, conv_eta, total_steps, n_out=10):
    chans = ARMS[arm]["channels"]
    layers, c_in = [], 1
    for li, k in enumerate(chans):
        layers.append(ConvLayer(k=k, c_in=c_in, seed=seed * 100 + li))
        c_in = k
    feat = _feat_dim(layers, side, side)
    head = build_graph([feat, 32, n_out], density=0.5, seed=seed)
    init_weights(head, seed=seed)
    model = DeepConvModel(layers, head, side, side)
    tr = DeepConvTrainer(_head_config(), model, None, None, seed=seed,
                         conv_eta=conv_eta, conv_eta_schedule="cosine",
                         total_steps=total_steps)
    return tr, model


def run_one(arm, seed, args):
    Xtr, ytr, Xte, yte = get_dataset(args.dataset, seed, n_points=args.points)
    side = int(round(np.sqrt(Xtr.shape[1])))
    Xtr = Xtr.reshape(-1, side, side)
    Xte = Xte.reshape(-1, side, side)
    n_out = int(max(ytr.max(), yte.max())) + 1
    tr, model = _build(arm, seed, side, args.conv_eta, args.steps, n_out=n_out)
    tr.X, tr.y = Xtr, ytr
    cap = min(args.train_eval_cap, len(Xtr))
    eval_idx = np.sort(np.random.default_rng(seed + 7).choice(len(Xtr), cap, replace=False))
    Xtr_eval, ytr_eval = Xtr[eval_idx], ytr[eval_idx]

    rec_step, test_acc, train_acc, head_syn = [], [], [], []
    t0 = time.perf_counter()
    for s in range(args.steps):
        tr.step()
        if s % args.record_every == 0 or s == args.steps - 1:
            rec_step.append(s)
            test_acc.append(tr.accuracy(Xte, yte))
            train_acc.append(tr.accuracy(Xtr_eval, ytr_eval))
            head_syn.append(len(model.head.synapses))
    wall = time.perf_counter() - t0

    return {
        "arm": arm, "seed": seed, "rec_step": rec_step,
        "test_acc": test_acc, "train_acc": train_acc, "head_syn": head_syn,
        "final_test_acc": test_acc[-1], "max_test_acc": float(np.max(test_acc)),
        "final_train_acc": train_acc[-1],
        "feat_dim": model.feat_dim(), "conv_params": _conv_params(model.layers),
        "head_syn_end": len(model.head.synapses),
        "n_sleep": sum(1 for e in tr.events if e["type"] == "sleep"),
        "wall_time": wall,
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
        rows[a] = {
            "final_mean": float(np.mean(finals)), "final_std": float(np.std(finals)),
            "max_mean": float(np.mean([r["max_test_acc"] for r in rs])),
            "train_mean": float(np.mean([r["final_train_acc"] for r in rs])),
            "feat_dim": int(rs[0]["feat_dim"]),
            "conv_params": int(rs[0]["conv_params"]),
            "head_syn_end": float(np.mean([r["head_syn_end"] for r in rs])),
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

    plt.figure(figsize=(7, 4.5))
    for a in arms:
        rs = by[a]
        steps = rs[0]["rec_step"]
        m = np.mean([r["test_acc"] for r in rs], axis=0)
        plt.plot(steps, m, label=a)
    plt.xlabel("step"); plt.ylabel("test accuracy"); plt.legend(); plt.grid(alpha=.3)
    plt.title(args.name); plt.tight_layout()
    plt.savefig(os.path.join(dest, "acc_curves.png"), dpi=110); plt.close()

    with open(os.path.join(dest, "metrics.json"), "w") as f:
        json.dump({"args": vars(args), "rows": rows, "results": results}, f)

    lines = [f"# Deep Conv-SPROUT — {args.name}", "",
             f"- **Dataset:** {args.dataset}  |  **Seeds:** {args.seeds}  "
             f"|  **Steps:** {args.steps}  |  **Baseline:** {args.baseline}",
             "- **Each conv layer:** 3x3 filters + ReLU + 2x2 maxpool, gradient-as-"
             "currency (meter/confidence/cosine consolidation); shared phasic head.",
             "", "## Results (mean ± std across seeds)", "",
             "| Arm | final test acc | max test acc | train acc | head feat | conv params | head syn | wall s | verdict |",
             "|---|---|---|---|---|---|---|---|---|"]
    for a in arms:
        rw = rows[a]
        lines.append(
            f"| {a} | {rw['final_mean']:.3f} ± {rw['final_std']:.3f} | "
            f"{rw['max_mean']:.3f} | {rw['train_mean']:.3f} | {rw['feat_dim']} | "
            f"{rw['conv_params']} | {rw['head_syn_end']:.0f} | {rw['wall']:.0f} | "
            f"{rw['verdict']} |")
    lines += ["", "Verdict = 95% seed-bootstrap CI of the final-test-acc "
              "difference vs baseline (UP/DOWN/~).", "", "![acc](acc_curves.png)", ""]
    with open(os.path.join(dest, "README.md"), "w") as f:
        f.write("\n".join(lines))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="depth1,depth2")
    ap.add_argument("--baseline", default="depth1")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--dataset", default="mnist-full")
    ap.add_argument("--points", type=int, default=12000)
    ap.add_argument("--train-eval-cap", type=int, default=2000)
    ap.add_argument("--record-every", type=int, default=1000)
    ap.add_argument("--conv-eta", type=float, default=0.02)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--name", default="deep-conv-run")
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
        print(f"  {a:14s} final {rw['final_mean']:.3f}±{rw['final_std']:.3f}  "
              f"max {rw['max_mean']:.3f}  feat {rw['feat_dim']}  "
              f"conv_params {rw['conv_params']}  {rw['verdict']}")
    return rows


if __name__ == "__main__":
    main()
