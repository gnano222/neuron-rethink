"""Cross-seed aggregation and the bootstrap significance verdict.

Single runs can't tell signal from noise, so the scorecard is built from N seeds
per variant. For each metric we report mean / std / a 95% CI, and against the
baseline variant we bootstrap the distribution of the mean *difference* (resample
seeds with replacement) to decide better / worse / no-clear-difference.

Pure NumPy — no scipy dependency.
"""

from __future__ import annotations

import math

import numpy as np

N_BOOT = 10000


def _finite(values) -> bool:
    return all(math.isfinite(v) for v in values)


def summarize(values, n_boot: int = N_BOOT, rng=None) -> dict:
    """mean / std / 95% bootstrap CI of the mean across seeds.

    If any value is non-finite (e.g. a threshold never reached → ``inf``), the
    mean reflects it and the CI is ``nan`` (a CI is meaningless then).
    """
    arr = np.asarray(list(values), dtype=float)
    if not _finite(arr):
        # mean of inf is inf (well-defined); std/CI are meaningless here
        return {"mean": float(arr.mean()), "std": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"),
                "n": int(arr.size)}
    mean = float(arr.mean())
    std = float(arr.std())
    rng = rng if rng is not None else np.random.default_rng(0)
    boot = np.array([rng.choice(arr, size=arr.size, replace=True).mean()
                     for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean": mean, "std": std, "ci_low": float(lo), "ci_high": float(hi),
            "n": int(arr.size)}


def verdict(ci_low: float, ci_high: float, direction: str) -> str:
    """Verdict for a (variant - baseline) difference CI, given metric direction.

    direction: ``"higher"`` (higher is better), ``"lower"``, or ``"neutral"``.
    """
    if direction == "neutral":
        return "≈"
    if ci_low > 0 and ci_high > 0:               # variant clearly above baseline
        return "better" if direction == "higher" else "worse"
    if ci_low < 0 and ci_high < 0:               # variant clearly below baseline
        return "better" if direction == "lower" else "worse"
    return "≈"                                    # CI straddles 0


def bootstrap_diff(variant_values, baseline_values, direction: str,
                   n_boot: int = N_BOOT, rng=None) -> dict:
    """Bootstrap CI of ``mean(variant) - mean(baseline)`` + a verdict."""
    v = np.asarray(list(variant_values), dtype=float)
    b = np.asarray(list(baseline_values), dtype=float)
    if not (_finite(v) and _finite(b)):
        # e.g. a threshold never reached (inf) in both arms -> inf - inf = nan;
        # the difference is undefined, so report n/a without the noisy warning
        with np.errstate(invalid="ignore"):
            diff_mean = float(v.mean() - b.mean())
        return {"diff_mean": diff_mean, "ci_low": float("nan"),
                "ci_high": float("nan"), "verdict": "n/a"}
    diff_mean = float(v.mean() - b.mean())
    rng = rng if rng is not None else np.random.default_rng(0)
    diffs = np.array([
        rng.choice(v, size=v.size, replace=True).mean()
        - rng.choice(b, size=b.size, replace=True).mean()
        for _ in range(n_boot)
    ])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff_mean": diff_mean, "ci_low": float(lo), "ci_high": float(hi),
            "verdict": verdict(lo, hi, direction)}


def aggregate_suite(results, baseline: str, directions: dict,
                    n_boot: int = N_BOOT, rng=None) -> dict:
    """Aggregate a list of per-run results into a scorecard structure.

    ``results``: list of ``{"variant", "seed", "final": {metric: value}}``.
    Returns ``{"variants", "baseline", "metrics": {metric: {variant: {...}}}}``
    where each non-baseline cell also carries ``diff_mean`` / CI / ``verdict``.
    """
    rng = rng if rng is not None else np.random.default_rng(0)

    variants = sorted({r["variant"] for r in results})
    by_variant: dict[str, list] = {v: [] for v in variants}
    for r in results:
        by_variant[r["variant"]].append(r)

    all_metrics: list[str] = []
    for r in results:
        for m in r["final"]:
            if m not in all_metrics:
                all_metrics.append(m)

    def values(variant, metric):
        return [r["final"][metric] for r in by_variant[variant]
                if metric in r["final"]]

    metrics: dict = {}
    for m in all_metrics:
        direction = directions.get(m, "neutral")
        metrics[m] = {}
        base_vals = values(baseline, m) if baseline in by_variant else []
        for v in variants:
            cell = summarize(values(v, m), n_boot=n_boot, rng=rng)
            if v != baseline and base_vals:
                # namespace the diff fields so they don't clobber the variant's
                # own mean-CI (ci_low/ci_high) from summarize
                d = bootstrap_diff(values(v, m), base_vals, direction,
                                   n_boot=n_boot, rng=rng)
                cell["diff_mean"] = d["diff_mean"]
                cell["diff_ci_low"] = d["ci_low"]
                cell["diff_ci_high"] = d["ci_high"]
                cell["verdict"] = d["verdict"]
            metrics[m][v] = cell

    return {"variants": variants, "baseline": baseline,
            "directions": directions, "metrics": metrics}
