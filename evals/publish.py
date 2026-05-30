"""Package one evaluation run into a git-committable, phone-browsable folder.

After a run writes its artifacts to ``output/eval/<...>`` (gitignored), this
copies the charts + data into ``docs/eval-runs/<run-id>/`` (tracked) alongside a
``README.md`` that embeds a curated key-metrics table, the full scorecard, and
the charts inline — so it renders as a single page on GitHub mobile.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time

from evals import report
from evals.metrics import METRIC_DIRECTIONS  # noqa: F401  (kept for parity/use)


# the curated headline metrics shown at the top of each run + in chat replies
KEY_METRICS = (
    "final_test_acc", "pre_shift_test_acc", "recovered_test_acc",
    "auc_test_acc", "max_grows_into_one_neuron", "oscillation_frac",
    "freeloader_frac", "conf_utility_corr", "dead_unit_count",
)

# data files copied verbatim from the run output dir (charts handled separately)
_DATA_FILES = ("scorecard.csv", "metrics.json", "summary.txt")


def build_highlight_table(agg, keys=KEY_METRICS) -> str:
    """A compact markdown table of just the headline metrics (present ones)."""
    variants = agg["variants"]
    baseline = agg["baseline"]
    head = ["Metric"] + [v + (" (baseline)" if v == baseline else "")
                         for v in variants]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "---|" * (len(variants) + 1)]
    for k in keys:
        if k not in agg["metrics"]:
            continue
        cells = [report._cell_text(k, agg["metrics"][k].get(v), v == baseline)
                 for v in variants]
        lines.append(f"| {report._label(k)} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def git_short_sha() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip()
    except Exception:
        return ""


def build_run_readme(agg, meta, image_names) -> str:
    L = [f"# Evaluation run: {meta.get('run_name', '')}", ""]
    L.append(f"- **Date:** {meta.get('date', '')}")
    L.append(f"- **Variants:** {', '.join(agg['variants'])} "
             f" (baseline: {agg['baseline']})")
    L.append(f"- **Seeds:** {meta.get('seeds', '')}  |  "
             f"**Dataset:** {meta.get('dataset', '')}  |  "
             f"**Steps:** {meta.get('steps', '')} (+{meta.get('shift', 0)} shift)")
    if meta.get("git_sha"):
        L.append(f"- **Commit:** {meta['git_sha']}")
    if meta.get("command"):
        L.append(f"- **Command:** `{meta['command']}`")
    L += ["", "## Key metrics", "", build_highlight_table(agg),
          "## Full scorecard", "", report.build_markdown(agg), "## Charts", ""]
    for name in image_names:
        title = name[:-4] if name.endswith(".png") else name
        L += [f"### {title}", f"![{title}]({name})", ""]
    return "\n".join(L) + "\n"


def publish_run(out_dir, agg, spec, dest_root="docs/eval-runs", run_id=None,
                command=None, date=None) -> str:
    """Copy a run's charts/data into ``dest_root/run_id`` and write its README.

    Returns the destination directory.
    """
    run_id = run_id or os.path.basename(os.path.normpath(out_dir))
    dest = os.path.join(dest_root, run_id)
    os.makedirs(dest, exist_ok=True)

    image_names = sorted(f for f in os.listdir(out_dir) if f.endswith(".png"))
    for f in image_names:
        shutil.copy2(os.path.join(out_dir, f), os.path.join(dest, f))
    for f in _DATA_FILES:
        src = os.path.join(out_dir, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dest, f))

    meta = {
        "run_name": run_id,
        "date": date or time.strftime("%Y-%m-%d %H:%M:%S"),
        "seeds": spec.seeds,
        "dataset": spec.dataset,
        "steps": spec.steps,
        "shift": spec.shift_steps,
        "git_sha": git_short_sha(),
        "command": command,
    }
    with open(os.path.join(dest, "README.md"), "w") as f:
        f.write(build_run_readme(agg, meta, image_names))
    return dest
