"""Render an aggregated suite into a scorecard (markdown / CSV / stdout table)
and a set of diagnostic plots.

The string builders are pure (tested against a synthetic aggregate). The plots
use matplotlib with the Agg backend so they work headless.
"""

from __future__ import annotations

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

from evals.metrics import METRIC_FAMILIES, METRIC_DIRECTIONS  # noqa: E402

_DIR_ARROW = {"higher": "↑", "lower": "↓", "neutral": ""}
_VERDICT_MARK = {"better": "▲", "worse": "▼", "≈": "≈", "n/a": "?", "": ""}


def _fmt(x) -> str:
    if x is None:
        return "—"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(xf):
        return "—"
    if math.isinf(xf):
        return "∞"
    if abs(xf) >= 1000 or xf == int(xf):
        return f"{xf:.0f}"
    return f"{xf:.3f}"


def verdict_marker(verdict: str) -> str:
    return _VERDICT_MARK.get(verdict, "")


def _present_metrics(agg) -> list:
    """Metrics actually in the aggregate, ordered by family then schema order."""
    out = []
    for _family, keys in METRIC_FAMILIES.items():
        for k in keys:
            if k in agg["metrics"]:
                out.append(k)
    return out


def _cell_text(metric, cell, is_baseline) -> str:
    if cell is None:
        return "—"
    base = f"{_fmt(cell.get('mean'))} ± {_fmt(cell.get('std'))}"
    if is_baseline:
        return base
    mark = verdict_marker(cell.get("verdict", ""))
    return f"{base} {mark}".strip()


def _label(metric) -> str:
    arrow = _DIR_ARROW.get(METRIC_DIRECTIONS.get(metric, "neutral"), "")
    return f"{metric} {arrow}".strip()


# -- markdown ----------------------------------------------------------------

def build_markdown(agg) -> str:
    variants = agg["variants"]
    baseline = agg["baseline"]
    head = ["Metric"] + [v + (" (baseline)" if v == baseline else "")
                          for v in variants]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "---|" * (len(variants) + 1)]
    empty = " |" * len(variants)
    for family, keys in METRIC_FAMILIES.items():
        present = [k for k in keys if k in agg["metrics"]]
        if not present:
            continue
        lines.append(f"| **{family}** |{empty}")
        for k in present:
            cells = [_cell_text(k, agg["metrics"][k].get(v), v == baseline)
                     for v in variants]
            lines.append(f"| {_label(k)} | " + " | ".join(cells) + " |")
    note = (f"\nBaseline: **{baseline}**. ▲ better / ▼ worse / ≈ no clear "
            "difference vs baseline (95% bootstrap CI of the mean difference). "
            "Cells show mean ± std across seeds.\n")
    return "\n".join(lines) + "\n" + note


# -- csv (long / tidy form) --------------------------------------------------

def build_csv(agg) -> str:
    cols = ["family", "metric", "direction", "variant", "mean", "std",
            "ci_low", "ci_high", "verdict"]
    rows = [",".join(cols)]
    for family, keys in METRIC_FAMILIES.items():
        for k in keys:
            if k not in agg["metrics"]:
                continue
            direction = agg["directions"].get(k, "neutral")
            for v in agg["variants"]:
                cell = agg["metrics"][k].get(v, {})
                rows.append(",".join([
                    family, k, direction, v,
                    repr(cell.get("mean", "")),
                    repr(cell.get("std", "")),
                    repr(cell.get("ci_low", "")),
                    repr(cell.get("ci_high", "")),
                    cell.get("verdict", ""),
                ]))
    return "\n".join(rows) + "\n"


# -- plain-text table (stdout) -----------------------------------------------

def build_table(agg) -> str:
    variants = agg["variants"]
    present = _present_metrics(agg)
    label_w = max([len("metric")] + [len(_label(k)) for k in present]) + 1
    col_w = max(16, max((len(v) for v in variants), default=8) + 2)
    head = "metric".ljust(label_w) + "".join(v.rjust(col_w) for v in variants)
    lines = [head, "-" * len(head)]
    for k in present:
        row = _label(k).ljust(label_w)
        for v in variants:
            cell = agg["metrics"][k].get(v, {})
            txt = _fmt(cell.get("mean"))
            if v != agg["baseline"] and cell.get("verdict"):
                txt += verdict_marker(cell["verdict"])
            row += txt.rjust(col_w)
        lines.append(row)
    return "\n".join(lines) + "\n"


# -- plots -------------------------------------------------------------------

def _by_variant(results) -> dict:
    out: dict = {}
    for r in results:
        out.setdefault(r["variant"], []).append(r)
    return out


def _stack(runs, key):
    arrs = [np.asarray(r["series"][key], dtype=float) for r in runs]
    n = min(len(a) for a in arrs)
    return np.array([a[:n] for a in arrs]), n


def _band_plot(ax, runs, key, label):
    M, n = _stack(runs, key)
    rec = np.asarray(runs[0]["series"]["rec_step"], dtype=float)[:n]
    mean, std = M.mean(0), M.std(0)
    line, = ax.plot(rec, mean, label=label)
    ax.fill_between(rec, mean - std, mean + std, alpha=0.2, color=line.get_color())
    si = runs[0].get("shift_start_index")
    if si is not None and si < n:
        ax.axvline(rec[si], color=line.get_color(), ls=":", lw=1, alpha=0.5)


def _plot_curves(by_variant, key, title, ylabel, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for variant, runs in by_variant.items():
        _band_plot(ax, runs, key, variant)
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def _plot_continual_curves(by_variant, path):
    """The forgetting curve: per-task A (solid) vs B (dashed) held-out accuracy,
    with vertical lines at each phase change. Task A visibly decays during phase
    B, then both recover during the A+B consolidation phase.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    for variant, runs in by_variant.items():
        A, n = _stack(runs, "test_accuracy_A")
        B, _ = _stack(runs, "test_accuracy_B")
        rec = np.asarray(runs[0]["series"]["rec_step"], dtype=float)[:n]
        line, = ax.plot(rec, A.mean(0), label=f"{variant} · A")
        ax.plot(rec, B.mean(0), ls="--", color=line.get_color(),
                label=f"{variant} · B")
    runs0 = next(iter(by_variant.values()))
    phase = runs0[0]["series"].get("phase", [])
    rec0 = runs0[0]["series"]["rec_step"]
    for i in range(1, min(len(phase), len(rec0))):
        if phase[i] != phase[i - 1]:
            ax.axvline(rec0[i], color="k", ls=":", lw=1, alpha=0.4)
    ax.set_title("continual: task A (solid) vs B (dashed); dotted = phase change")
    ax.set_xlabel("step")
    ax.set_ylabel("held-out accuracy")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def _plot_churn(by_variant, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for variant, runs in by_variant.items():
        grow, n = _stack(runs, "cum_grow")
        prune, _ = _stack(runs, "cum_prune")
        rec = np.asarray(runs[0]["series"]["rec_step"], dtype=float)[:n]
        total = (grow + prune).mean(0)
        ax.plot(rec, total, label=variant)
    ax.set_title("cumulative structural churn (grows + prunes)")
    ax.set_xlabel("step")
    ax.set_ylabel("events")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def _plot_quality(variant, runs, path):
    utils = np.concatenate([np.asarray(r["dist"]["utilities"], float)
                            for r in runs]) if runs else np.array([])
    confs = np.concatenate([np.asarray(r["dist"]["confidences"], float)
                            for r in runs]) if runs else np.array([])
    ages = np.concatenate([np.asarray(r["dist"]["ages"], float)
                           for r in runs]) if runs else np.array([])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    if utils.size:
        ax[0].hist(utils, bins=20, color="tab:blue", alpha=0.8)
    ax[0].set_title("utility distribution (survivors)")
    ax[0].set_xlabel("utility")
    if utils.size and confs.size:
        ax[1].scatter(utils, confs, s=8, alpha=0.4, color="tab:purple")
    ax[1].set_title("confidence vs utility")
    ax[1].set_xlabel("utility")
    ax[1].set_ylabel("confidence")
    if ages.size:
        ax[2].hist(ages, bins=20, color="tab:green", alpha=0.8)
    ax[2].set_title("survivor age distribution")
    ax[2].set_xlabel("age (steps)")
    fig.suptitle(f"synapse quality — {variant}")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def _plot_verdict_heatmap(agg, path):
    baseline = agg["baseline"]
    variants = [v for v in agg["variants"] if v != baseline]
    metrics_list = [k for k in _present_metrics(agg)
                    if METRIC_DIRECTIONS.get(k, "neutral") != "neutral"]
    if not variants or not metrics_list:
        # nothing to compare (e.g. baseline only); still emit a placeholder
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, "no comparison\n(baseline only)", ha="center",
                va="center")
        ax.axis("off")
        fig.savefig(path, dpi=90)
        plt.close(fig)
        return
    score = {"better": 1.0, "worse": -1.0, "≈": 0.0, "n/a": 0.0, "": 0.0}
    grid = np.array([[score.get(agg["metrics"][k][v].get("verdict", ""), 0.0)
                      for k in metrics_list] for v in variants])
    fig, ax = plt.subplots(figsize=(max(6, len(metrics_list) * 0.4),
                                    1.5 + 0.5 * len(variants)))
    ax.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels(variants)
    ax.set_xticks(range(len(metrics_list)))
    ax.set_xticklabels(metrics_list, rotation=90, fontsize=7)
    ax.set_title(f"verdict vs {baseline} (green=better, red=worse)")
    fig.savefig(path, dpi=90, bbox_inches="tight")
    plt.close(fig)


# -- orchestration -----------------------------------------------------------

def write_report(agg, results, out_dir) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "markdown": os.path.join(out_dir, "scorecard.md"),
        "csv": os.path.join(out_dir, "scorecard.csv"),
        "json": os.path.join(out_dir, "metrics.json"),
    }
    with open(paths["markdown"], "w") as f:
        f.write(build_markdown(agg))
    with open(paths["csv"], "w") as f:
        f.write(build_csv(agg))
    with open(paths["json"], "w") as f:
        json.dump({"aggregate": agg,
                   "runs": [{"variant": r["variant"], "seed": r["seed"],
                             "final": r["final"]} for r in results]},
                  f, indent=2)

    by_variant = _by_variant(results)
    if results and results[0].get("regime") == "continual":
        _plot_continual_curves(
            by_variant, os.path.join(out_dir, "continual_curves.png"))
    else:
        _plot_curves(by_variant, "test_accuracy", "test accuracy (dotted = shift)",
                     "accuracy", os.path.join(out_dir, "acc_curves.png"))
    _plot_curves(by_variant, "synapse_count", "synapse count", "synapses",
                 os.path.join(out_dir, "count_curves.png"))
    _plot_churn(by_variant, os.path.join(out_dir, "churn_curves.png"))
    _plot_verdict_heatmap(agg, os.path.join(out_dir, "verdict_heatmap.png"))
    for variant, runs in by_variant.items():
        _plot_quality(variant, runs,
                      os.path.join(out_dir, f"quality_{variant}.png"))
    return paths
