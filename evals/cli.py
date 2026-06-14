"""Command-line entry for the comparative evaluation harness.

Run a suite of variants across seeds (in parallel), aggregate with bootstrap
verdicts vs a baseline, and write a scorecard + diagnostic plots.

    python evaluate.py --variants phasic-startle-k4,phasic-startle --seeds 10 \
        --dataset spirals --steps 15000 --shift 3000 --baseline phasic-startle-k4
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from evals.spec import SuiteSpec, VARIANTS
from evals.runner import run_suite
from evals.aggregate import aggregate_suite
from evals.report import write_report, build_table
from evals.publish import publish_run
from evals.metrics import METRIC_DIRECTIONS

DEFAULT_LAYERS = (2, 16, 16, 16, 2)   # w16: the width-sweep sweet spot
CACHE_DIR = os.path.join("output", "eval", "cache")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="SPROUT comparative evaluation")
    ap.add_argument("--variants", default="phasic-startle-k4,phasic-startle",
                    help="comma-separated variant names")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--dataset", default="spirals",
                    choices=["spirals", "blobs", "digits", "mnist14"])
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--shift", type=int, default=0,
                    help="concept-shift (label-swap) steps after the main run")
    ap.add_argument("--regime", default="single", choices=["single", "continual"],
                    help="single (task + optional label-swap) or continual "
                         "(A->B->A+B forgetting benchmark, two offset spirals)")
    ap.add_argument("--steps-a", type=int, default=15000,
                    help="continual phase A steps (learn the left spiral)")
    ap.add_argument("--steps-b", type=int, default=15000,
                    help="continual phase B steps (right spiral only; A erodes)")
    ap.add_argument("--steps-ab", type=int, default=10000,
                    help="continual phase A+B steps (interleaved consolidation)")
    ap.add_argument("--continual-turns", type=float, default=0.6,
                    help="continual: spiral turns (gentler => the 4-arm union "
                         "stays learnable, so consolidation has headroom)")
    ap.add_argument("--baseline", default="phasic-startle-k4")
    ap.add_argument("--jobs", type=int, default=None,
                    help="parallel workers (default: cpu count)")
    ap.add_argument("--record-every", type=int, default=200)
    ap.add_argument("--density", type=float, default=0.4)
    ap.add_argument("--points", type=int, default=600)
    ap.add_argument("--layers", default=None, help="e.g. 2,10,10,8,2")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--publish", action="store_true",
                    help="package this run into docs/eval-runs/<run-name>/ "
                         "(git-committable, renders on GitHub mobile)")
    ap.add_argument("--run-name", default=None,
                    help="name for the published run folder (default: out basename)")
    ap.add_argument("--publish-dir", default="docs/eval-runs",
                    help="root dir for published run folders")
    return ap.parse_args(argv)


def build_spec(args) -> SuiteSpec:
    layers = (tuple(int(x) for x in args.layers.split(","))
              if args.layers else DEFAULT_LAYERS)
    variants = tuple(v.strip() for v in args.variants.split(",") if v.strip())
    return SuiteSpec(
        variants=variants, seeds=args.seeds, dataset=args.dataset,
        steps=args.steps, shift_steps=args.shift,
        record_every=args.record_every, baseline=args.baseline,
        layers=layers, density=args.density, n_points=args.points,
        regime=args.regime, steps_a=args.steps_a, steps_b=args.steps_b,
        steps_ab=args.steps_ab, continual_turns=args.continual_turns,
    )


def main(argv=None):
    args = parse_args(argv)
    spec = build_spec(args)

    unknown = [v for v in spec.variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variant(s): {', '.join(unknown)}; "
                         f"known: {', '.join(sorted(VARIANTS))}")
    if spec.baseline not in spec.variants:
        raise SystemExit(f"baseline {spec.baseline!r} must be one of the "
                         f"variants ({', '.join(spec.variants)})")

    out = args.out or os.path.join(
        "output", "eval", f"{spec.dataset}_{time.strftime('%Y%m%d_%H%M%S')}")
    cache_dir = None if args.no_cache else CACHE_DIR

    if spec.regime == "continual":
        steps_desc = f"A->B->A+B {spec.steps_a}+{spec.steps_b}+{spec.steps_ab} steps"
    else:
        steps_desc = f"{spec.steps}+{spec.shift_steps} steps"
    print(f"running {len(spec.variants)} variant(s) x {spec.seeds} seed(s) "
          f"= {len(spec.variants) * spec.seeds} runs ({steps_desc} each) ...")
    results = run_suite(spec, jobs=args.jobs, cache_dir=cache_dir,
                        use_cache=not args.no_cache)
    agg = aggregate_suite(results, baseline=spec.baseline,
                          directions=METRIC_DIRECTIONS, n_boot=args.n_boot,
                          rng=np.random.default_rng(0))
    write_report(agg, results, out)

    table = build_table(agg)
    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write(table)
    print("\n" + table)
    print("artifacts ->", out)

    if args.publish:
        raw = argv if argv is not None else sys.argv[1:]
        command = "python evaluate.py " + " ".join(raw)
        run_id = args.run_name or os.path.basename(os.path.normpath(out))
        dest = publish_run(out, agg, spec, dest_root=args.publish_dir,
                           run_id=run_id, command=command)
        print("published ->", dest)
    return agg


if __name__ == "__main__":
    main()
