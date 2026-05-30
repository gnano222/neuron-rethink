---
name: running-sprout-evals
description: Use when running, comparing, reporting, or interpreting a SPROUT evaluation — evaluate.py, the evals/ harness, comparing network variants (currency vs legacy-full vs core), or any "run an eval / compare variants" request. Especially when results must be readable from mobile.
---

# Running SPROUT evaluations

## Overview

Every eval run must be consumable on a phone: a **key-metrics table in the chat
reply** AND a **git-committed per-run folder** that renders as one page on
GitHub. Reliability comes from `--publish` (deterministic packaging), not from
remembering to copy files.

## Protocol (follow every run)

1. **Run with `--publish`** (packages a committable run folder automatically):
   ```
   python evaluate.py --variants currency,legacy-full --seeds 5 \
     --dataset spirals --steps 30000 --shift 6000 --baseline legacy-full \
     --jobs 6 --no-cache --publish --run-name <name>
   ```
   Writes `output/eval/<...>` (gitignored) **and** `docs/eval-runs/<name>/`
   (tracked): `README.md` (key-metrics table + full scorecard + charts inline),
   all charts, `scorecard.csv`, `metrics.json`, `summary.txt`.
   Use `--no-cache` after any metric-schema change (the cache key does not
   version the schema).

2. **Commit + push the run folder** (pre-authorized — do not re-ask):
   ```
   git add docs/eval-runs/<name> && git commit -m "eval: <name>" && git push
   ```

3. **Write the chat reply** with ALL of:
   - the **key-metrics table** (curated headline rows with ▲/▼/≈), NOT the full
     ~35-row dump. **Include the "What it means" description column** (the
     `build_highlight_table` output already has it) — TEMPORARY at the user's
     request while they learn the schema; drop the column only when they say so.
   - a 2–3 sentence plain-language verdict
   - the path `docs/eval-runs/<name>/README.md` so it opens on phone
   - honest **wins AND losses** — never bury a regression

## What lands in `docs/eval-runs/<name>/`

| File | Use |
|---|---|
| `README.md` | the phone view: metadata + key metrics + scorecard + charts |
| `acc_curves.png`, `verdict_heatmap.png`, `count_curves.png`, `churn_curves.png`, `quality_*.png` | diagnostics |
| `scorecard.csv`, `metrics.json` | data for re-analysis |
| `summary.txt` | the plain stdout table |

## Reading the numbers (don't over-claim)

- Cells are **mean ± std across seeds**; ▲/▼/≈ is a **95% bootstrap CI of the
  difference vs baseline**. `≈` = CI straddles 0 → **no clear difference, do not
  claim a win**. `?` = n/a (e.g. a threshold never reached → `∞`).
- `↑`/`↓` on a metric = higher/lower is better. Default **5 seeds**; one lucky
  seed is exactly what this harness exists to catch.

Headline (KEY) metrics: `final_test_acc`, `pre_shift_test_acc`,
`recovered_test_acc`, `auc_test_acc`, `max_grows_into_one_neuron`,
`oscillation_frac`, `freeloader_frac`, `conf_utility_corr`, `dead_unit_count`.

## Common mistakes

- Forgetting `--publish` → nothing reaches the phone.
- Forgetting to push → folder exists locally only.
- Dumping all metrics instead of the curated key table.
- Claiming a win on a `≈` verdict or a single noisy seed.
