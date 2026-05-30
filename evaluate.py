"""Top-level entry for the SPROUT comparative evaluation harness.

Thin wrapper around :func:`evals.cli.main`. See ``--help`` for options, or the
design at docs/superpowers/specs/2026-05-30-sprout-eval-harness-design.md.

    python evaluate.py --variants currency,legacy-full,core --seeds 10 \
        --dataset spirals --steps 30000 --shift 6000 --jobs 8
"""

from __future__ import annotations

from evals.cli import main

if __name__ == "__main__":
    main()
