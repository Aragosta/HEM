#!/usr/bin/env python3
"""T1: how many experts should be active, and is depth a substitute for them?

Two literatures meet here and neither has tested the intersection.

**From MoE theory.** *Sparse Mixture-of-Experts for Compositional
Generalization* (arXiv:2410.13964) derives a scaling law by balancing
approximation against estimation error and concludes that the optimal number of
*activated* experts sits strictly between minimal (1-2) and full activation,
growing with task complexity. Everything in E0-E4 was run at k=2 of 16 without
ever asking whether that was right.

**From our own E1.** Depth turned out to be a *substitute* for data rather than
a complement: at 2 hops, doubling the budget erased depth's advantage entirely.
Depth and active-expert count both multiply per-token FLOPs, so the obvious next
question is whether they are substitutes for each other too.

The grid answers both at once: ``k`` in {1,2,4,8} x hop count in {2,3} x
``loops`` in {1,2}. Hop count is the complexity knob the MoE scaling law needs;
loops is the axis E1 established; k is the axis nobody varied.

Registered predictions:

- **T1a.** k* grows with hop count (2410.13964's claim, on a task with an
  explicit complexity parameter rather than a benchmark proxy).
- **T1b.** k and R are partial substitutes: at matched FLOPs, (R=2, k=2) beats
  (R=1, k=4) on composition. If instead they are complements, the iso-compute
  frontier has an interior optimum and the "loops need width" story is wrong.
- **T1c.** k* > 2 at 3 hops. If k*=2 everywhere, the scaling law does not reach
  down to this scale and T3/T4 are measuring a knob that does not matter.

Usage::

    python t1_sparsity_depth.py --seeds 0,1 --workers 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import load_results, mean_sd, run_jobs                # noqa: E402
from spec import BATCH, HOPS_CFG, LR, hop_spec                    # noqa: E402

KS = (1, 2, 4, 8)
HOPS = (2, 3)
LOOPS = (1, 2)
STEPS = 2000


def jobs(seeds, steps: int = STEPS):
    out = []
    for h in HOPS:
        for r in LOOPS:
            for k in KS:
                for seed in seeds:
                    out.append({
                        "name": f"t1_h{h}_R{r}_k{k}_s{seed}", "task": "hops",
                        "spec": hop_spec(h),
                        "cfg": {**HOPS_CFG, "loops": r, "n_active": k},
                        "steps": steps, "batch": BATCH, "lr": LR, "seed": seed})
    return out


def report() -> str:
    lines = ["# T1 -- active experts x task complexity x depth", ""]
    for h in HOPS:
        lines += [f"## {h}-hop composition (accuracy)", "",
                  "| loops | " + " | ".join(f"k={k}" for k in KS) + " | best k |",
                  "| --- | " + " | ".join("---" for _ in KS) + " | --- |"]
        for r in LOOPS:
            cells, means = [], {}
            for k in KS:
                runs = load_results(f"t1_h{h}_R{r}_k{k}_s")
                if not runs:
                    cells.append("-")
                    continue
                m, sd = mean_sd([x["final"]["acc"] for x in runs.values()])
                means[k] = m
                cells.append(f"{m:.3f} ± {sd:.3f}")
            best = max(means, key=means.get) if means else "-"
            lines.append(f"| {r} | " + " | ".join(cells) + f" | **{best}** |")
        lines.append("")

    lines += ["## iso-compute: is depth a substitute for active experts?", "",
              "Cells with comparable forward FLOPs per token, so the question is "
              "where a fixed budget should go.", "",
              "| hops | arm | FLOPs/token | accuracy |",
              "| --- | --- | --- | --- |"]
    for h in HOPS:
        for label, r, k in (("R=1, k=4", 1, 4), ("R=2, k=2", 2, 2),
                            ("R=1, k=8", 1, 8), ("R=2, k=4", 2, 4)):
            runs = load_results(f"t1_h{h}_R{r}_k{k}_s")
            if not runs:
                continue
            m, sd = mean_sd([x["final"]["acc"] for x in runs.values()])
            flops = list(runs.values())[0]["flops_per_token"]
            lines.append(f"| {h} | {label} | {flops:.2e} | {m:.3f} ± {sd:.3f} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--workers", type=int, default=4)
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
