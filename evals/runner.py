"""Run driver: train one (variant, seed), snapshot metrics over time, and
aggregate a whole suite across seeds in parallel with a disk cache.

The harness owns its own minimal training loop (mirroring ``run.run`` without
rendering) so it can measure a held-out test set and quality metrics at each
record step. It never mutates ``sprout/``.
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import numpy as np

# Defensive: keep BLAS single-threaded per worker process. The hand-rolled
# forward/backward are Python scalar loops (no BLAS), so this is essentially a
# no-op today, but it prevents oversubscription if that ever changes. Must be
# set before heavy numeric imports in spawned workers — module import time is.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from sprout.network import build_graph, init_weights      # noqa: E402
from sprout.train import Trainer, accuracy                # noqa: E402
from sprout.data import generate_blobs, generate_spirals  # noqa: E402
from evals.spec import make_config                         # noqa: E402
from evals import metrics                                  # noqa: E402


SERIES_KEYS = ("rec_step", "train_accuracy", "train_loss", "test_accuracy",
               "test_loss", "synapse_count", "mean_confidence",
               "cum_grow", "cum_prune")


def _gen(dataset, n, seed, turns, noise):
    if dataset == "blobs":
        return generate_blobs(n=n, seed=seed)
    return generate_spirals(n=n, seed=seed, turns=turns, noise=noise)


def _snapshot(series, net, tr, X_tr, y_tr, X_te, y_te):
    """Append one timepoint to the series (cheap metrics only)."""
    series["rec_step"].append(tr.step_idx)
    series["train_accuracy"].append(accuracy(net, X_tr, y_tr))
    series["train_loss"].append(metrics.test_loss(net, X_tr, y_tr))
    series["test_accuracy"].append(accuracy(net, X_te, y_te))
    series["test_loss"].append(metrics.test_loss(net, X_te, y_te))
    series["synapse_count"].append(len(net.synapses))
    confs = [s.confidence for s in net.synapses.values()]
    series["mean_confidence"].append(float(np.mean(confs)) if confs else 0.0)
    series["cum_grow"].append(sum(1 for e in tr.events if e["type"] == "grow"))
    series["cum_prune"].append(sum(1 for e in tr.events if e["type"] == "prune"))


def run_one(variant_name, seed, spec):
    """Train one (variant, seed) and return a plain-dict RunResult."""
    cfg = make_config(variant_name)
    X_tr, y_tr = _gen(spec.dataset, spec.n_points, seed, spec.turns, spec.noise)
    X_te, y_te = _gen(spec.dataset, spec.n_points, seed + spec.test_seed_offset,
                      spec.turns, spec.noise)

    net = build_graph(list(spec.layers), density=spec.density, seed=seed)
    init_weights(net, seed=seed)
    tr = Trainer(cfg, net, X_tr, y_tr, seed=seed)
    initial_edges = set(net.synapses)
    series = {k: [] for k in SERIES_KEYS}

    def loop(n_steps, y_tr_cur, y_te_cur):
        for s in range(n_steps):
            record = (s % spec.record_every == 0) or (s == n_steps - 1)
            tr.step(record=False)
            if record:
                _snapshot(series, net, tr, X_tr, y_tr_cur, X_te, y_te_cur)

    loop(spec.steps, y_tr, y_te)

    shift_start_index = None
    y_te_final = y_te
    if spec.shift_steps > 0:
        shift_start_index = len(series["rec_step"])
        y_tr_sw, y_te_final = 1 - y_tr, 1 - y_te
        tr.X, tr.y = X_tr, y_tr_sw            # concept shift: swap the labels
        loop(spec.shift_steps, y_tr_sw, y_te_final)

    final, dist = metrics.final_snapshot(
        net, X_te, y_te_final, tr.events, initial_edges, series,
        shift_start_index, cfg)

    return {
        "variant": variant_name,
        "seed": seed,
        "config": asdict(cfg),
        "series": series,
        "final": final,
        "dist": dist,
        "shift_start_index": shift_start_index,
        "initial_edge_count": len(initial_edges),
    }


# -- caching -----------------------------------------------------------------

def _cache_key(variant_name, seed, spec) -> str:
    payload = {
        "variant": variant_name,
        "config": asdict(make_config(variant_name)),
        "seed": seed,
        "dataset": spec.dataset,
        "steps": spec.steps,
        "shift_steps": spec.shift_steps,
        "record_every": spec.record_every,
        "layers": list(spec.layers),
        "density": spec.density,
        "n_points": spec.n_points,
        "turns": spec.turns,
        "noise": spec.noise,
        "test_seed_offset": spec.test_seed_offset,
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def _run_one_cached(variant_name, seed, spec, cache_dir, use_cache):
    path = None
    if cache_dir:
        path = os.path.join(cache_dir, _cache_key(variant_name, seed, spec) + ".json")
        if use_cache and os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    res = run_one(variant_name, seed, spec)
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump(res, f)
    return res


def _job(args):
    """Top-level (picklable) worker entry for the process pool."""
    return _run_one_cached(*args)


def run_suite(spec, jobs=None, cache_dir=None, use_cache=True):
    """Run every (variant, seed) in ``spec`` and return the list of RunResults."""
    jobs = jobs or os.cpu_count() or 1
    tasks = [(v, seed, spec, cache_dir, use_cache)
             for v in spec.variants for seed in spec.seed_list()]
    if jobs == 1:
        return [_job(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        return list(ex.map(_job, tasks))
