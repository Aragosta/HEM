#!/usr/bin/env python3
"""T2b: the trajectory runs redone at R=4, because R=2 could not answer T2.

A design error worth recording. T1, T3 and T4 all ran at R=2 to keep the grid
affordable, which leaves the expert path with exactly **one** transition. T2's
questions -- does the expert set converge, does routing convergence track state
convergence, do wrong answers settle later -- are all questions about the
*shape* of a curve, and one point is not a curve. The T2 tables from that round
are reported, but they cannot be read as evidence either way.

This is the minimal fix: the same task at R=4, four arms, so there are three
transitions to look at. It also carries the corrected `k_by_loop` (the wide-entry
schedule was leaking into the prelude and coda, which are not part of the loop
and were quietly inflating the `phase` arm's compute).

| arm | what it adds |
| --- | --- |
| `base` | R=4, k=2 -- the trajectory baseline |
| `phase` | wide entry, narrow refinement, now core-only |
| `local` | threshold routing, so the k profile across four loops is visible |
| `bias` | E3's winning conditioner, to see what it does to the path |

Usage::

    python t2b_depth4.py --seeds 0,1 --workers 4
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
ARMS = {
    "base": {},
    "phase": {"k_by_loop": (8, 4, 2, 2)},
    "local": {"route_mode": "threshold"},
    "bias": {"step_routing": "bias"},
}


def jobs(seeds, steps: int = STEPS):
    return [{"name": f"t2b_{arm}_s{seed}", "task": "hops", "spec": hop_spec(HOP),
             "cfg": {**HOPS_CFG, "loops": 4, "n_active": 2, **delta},
             "steps": steps, "batch": BATCH, "lr": LR, "seed": seed}
            for arm, delta in ARMS.items() for seed in seeds]


def report() -> str:
    lines = ["# T2b -- expert trajectories at R=4", "",
             "## accuracy", "",
             "| arm | accuracy | sd | n |", "| --- | --- | --- | --- |"]
    for arm in ARMS:
        runs = load_results(f"t2b_{arm}_s")
        if not runs:
            continue
        m, sd = mean_sd([r["final"]["acc"] for r in runs.values()])
        lines.append(f"| {arm} | {m:.3f} | {sd:.3f} | {len(runs)} |")
    lines.append("")

    lines += ["## the convergence curve (T2a)", "",
              "Expert-set overlap between consecutive loops, and the relative "
              "size of the state update at the same transition. Under the "
              "fixed-point reading the first rises and the second falls, "
              "together.", "",
              "| arm | seed | transition | expert overlap | state move |",
              "| --- | --- | --- | --- | --- |"]
    for arm in ARMS:
        for name, r in sorted(load_results(f"t2b_{arm}_s").items()):
            traj = r.get("trajectories", {})
            ov = traj.get(f"overlap_h{HOP}") or []
            mv = traj.get(f"state_move_h{HOP}") or []
            for i, o in enumerate(ov):
                move = f"{mv[i]:.3f}" if i < len(mv) else "-"
                lines.append(f"| {arm} | {name[-1]} | {i + 1}->{i + 2} | "
                             f"{o:.3f} | {move} |")
    lines.append("")

    lines += ["## settle step, right versus wrong (T2b)", "",
              "| arm | correct | wrong | difference |",
              "| --- | --- | --- | --- |"]
    for arm in ARMS:
        right, wrong = [], []
        for r in load_results(f"t2b_{arm}_s").values():
            traj = r.get("trajectories", {})
            if f"settle_correct_h{HOP}" in traj and f"settle_wrong_h{HOP}" in traj:
                right.append(traj[f"settle_correct_h{HOP}"])
                wrong.append(traj[f"settle_wrong_h{HOP}"])
        if right:
            mr, _ = mean_sd(right)
            mw, _ = mean_sd(wrong)
            lines.append(f"| {arm} | {mr:.3f} | {mw:.3f} | {mw - mr:+.3f} |")
    lines.append("")

    lines += ["## experts per token by loop position (T4b, four loops)", "",
              "| arm | seed | loop 1 | loop 2 | loop 3 | loop 4 |",
              "| --- | --- | --- | --- | --- | --- |"]
    for arm in ARMS:
        for name, r in sorted(load_results(f"t2b_{arm}_s").items()):
            graph = r.get("expert_graph", {})
            if not graph:
                continue
            cells = [f"{graph[f'loop{i}']['mean_k']:.2f}"
                     if f"loop{i}" in graph else "-" for i in (1, 2, 3, 4)]
            lines.append(f"| {arm} | {name[-1]} | " + " | ".join(cells) + " |")
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
