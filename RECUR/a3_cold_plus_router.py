#!/usr/bin/env python3
"""A3: does the router bias still buy anything once the temperature is right?

A1 found the largest effect in this project and pointed the opposite way to my
prediction. Accuracy at R=4 runs 0.678 / 0.639 / 0.608 / 0.607 / 0.456 as the
temperature multiplier goes 0.25 / 0.5 / 1 / 2 / 4, so **colder is better** and
the default 1/sqrt(head_dim) is already on the sharp side of the optimum. At
x0.25 the plain model scores 0.678 -- which is within noise of the 0.671 that
E3's per-loop router bias earned at the default temperature.

That raises a question A2 could not answer, because A2 varied a *learned*
per-loop temperature (which failed on its own terms, -0.030). The clean version
is: fix the temperature at the value A1 says is best, then add the router bias
and see whether it still adds anything.

| arm | temperature | router |
| --- | --- | --- |
| `cold` | x0.25 (already run as `a1_R4_b0p25_*`) | plain |
| `cold_bias` | x0.25 | per-loop bias |

Registered predictions:

- **A3a.** `cold_bias` is within noise of `cold`. If a scalar per head and a
  table of per-loop router biases are two ways of buying the same thing, the
  cheaper one wins and E3's result is a temperature result wearing a routing
  costume.
- **A3b, the alternative.** `cold_bias` beats `cold` by roughly what `bias`
  beat `none` (+0.06). Then they are genuinely different mechanisms, the
  upstream/downstream story is wrong, and both belong in the architecture.

Usage::

    python a3_cold_plus_router.py --seeds 0,1 --workers 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import load_results, mean_sd, run_jobs                # noqa: E402
from spec import BATCH, HOPS_CFG, LR, hop_spec                    # noqa: E402

HOP = 3
STEPS = 2000
COLD = 0.25


def jobs(seeds, steps: int = STEPS):
    return [{"name": f"a3_coldbias_s{seed}", "task": "hops", "spec": hop_spec(HOP),
             "cfg": {**HOPS_CFG, "loops": 4, "beta_scale": COLD,
                     "step_routing": "bias"},
             "steps": steps, "batch": BATCH, "lr": LR, "seed": seed}
            for seed in seeds]


def report() -> str:
    rows = [("cold (x0.25, no bias)", load_results("a1_R4_b0p25_s")),
            ("cold + per-loop bias", load_results("a3_coldbias_s")),
            ("default temperature, no bias", load_results("a2_none_s")),
            ("default temperature + bias", load_results("a2_bias_s"))]
    lines = ["# A3 -- cold temperature, with and without the router bias", "",
             "| arm | accuracy | sd | n |", "| --- | --- | --- | --- |"]
    means = {}
    for label, runs in rows:
        if not runs:
            continue
        m, sd = mean_sd([r["final"]["acc"] for r in runs.values()])
        means[label] = m
        lines.append(f"| {label} | {m:.3f} | {sd:.3f} | {len(runs)} |")
    lines.append("")
    if len(means) == 4:
        cold, cold_bias = rows[0][0], rows[1][0]
        warm, warm_bias = rows[2][0], rows[3][0]
        lines += [
            f"- router bias at the default temperature: "
            f"{means[warm_bias] - means[warm]:+.3f}",
            f"- router bias at the cold temperature: "
            f"{means[cold_bias] - means[cold]:+.3f}",
            f"- temperature alone (x0.25 vs x1, no bias): "
            f"{means[cold] - means[warm]:+.3f}", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    if not args.report_only:
        run_jobs(jobs(seeds, args.steps), workers=args.workers,
                 dry_run=args.dry_run)
    if not args.dry_run:
        print(report())


if __name__ == "__main__":
    main()
