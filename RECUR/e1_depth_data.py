#!/usr/bin/env python3
"""E1: does the useful depth ceiling move with the token budget?

This is the brief's question 1 and the one on which the rest of the programme
turns. Huginn saturates with test-time depth at 800B tokens; Ouro trained at
4-8 recurrent steps and shipped 4. Both are consistent with two very different
worlds:

* the ceiling is a property of the **architecture** -- four steps is simply
  enough, and depth is a modest efficiency trick;
* the ceiling is a property of the **budget** -- the models were under-trained
  for the depth attempted, and useful depth keeps extending with data.

The grid separates them: ``loops`` in {1,2,4,8} crossed with three training
budgets. Two readings, both pre-registered in `DESIGN.md`:

**Reading A (the ceiling).** For each budget, the depth at which accuracy stops
improving. If that depth is the same at 1x and 4x data, the interaction is
absent at this scale.

**Reading B (the fixed-compute diagonal).** Cells with equal ``loops x steps``
cost roughly equal training FLOPs. Along a diagonal the question is not "does
depth help" but "is depth where a fixed budget should go", which is the
question an architect actually faces.

The composition task carries the depth hypothesis; the byte-level arm is the
control that says whether any gain is composition or just capacity. A gain on
bytes as large as on hops would mean the loop is buying something other than
sequential composition, and P2 would be wrong in an informative way.

Usage::

    python e1_depth_data.py --seeds 0,1 --workers 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import load_results, mean_sd, run_jobs                # noqa: E402
from spec import (BATCH, BYTE_BATCH, BYTE_CFG, BYTE_LR, BYTE_SEQ,
                  HOPS_CFG, LR, hop_spec)                         # noqa: E402

LOOPS = (1, 2, 4)
BUDGETS = (1500, 3000)              # 1x, 2x
HOPS = (2, 3)                       # how much composition an example needs
BYTE_LOOPS = (1, 2, 4)
BYTE_BUDGETS = (300, 600)


def jobs(seeds):
    out = []
    for h in HOPS:
        for r in LOOPS:
            for steps in BUDGETS:
                for seed in seeds:
                    out.append({
                        "name": f"e1_hops{h}_R{r}_b{steps}_s{seed}", "task": "hops",
                        "spec": hop_spec(h), "cfg": {**HOPS_CFG, "loops": r},
                        "steps": steps, "batch": BATCH, "lr": LR, "seed": seed,
                        # unrolling past the trained depth is free here and is
                        # the Huginn extrapolation claim, so it is recorded
                        "eval_loops": (1, 2, 4, 8)})
    for r in BYTE_LOOPS:
        for steps in BYTE_BUDGETS:
            for seed in seeds:
                out.append({
                    "name": f"e1_bytes_R{r}_b{steps}_s{seed}",
                    "task": "bytes:wikitext2", "cfg": {**BYTE_CFG, "loops": r},
                    "steps": steps, "batch": BYTE_BATCH, "seq_len": BYTE_SEQ,
                    "lr": BYTE_LR, "seed": seed})
    return out


def _grid(prefix, loops, budgets, metric):
    rows = {}
    for r in loops:
        for steps in budgets:
            runs = load_results(f"{prefix}_R{r}_b{steps}_s")
            if runs:
                rows[(r, steps)] = mean_sd([x["final"][metric] for x in runs.values()])
    return rows


def report() -> str:
    lines = ["# E1 -- depth x data", ""]

    for h in HOPS:
        grid = _grid(f"e1_hops{h}", LOOPS, BUDGETS, "acc")
        if not grid:
            continue
        lines += [f"## {h}-hop composition: accuracy by depth and budget", "",
                  "| loops | " + " | ".join(f"{b} steps" for b in BUDGETS) + " |",
                  "| --- | " + " | ".join("---" for _ in BUDGETS) + " |"]
        for r in LOOPS:
            cells = []
            for b in BUDGETS:
                m, sd = grid.get((r, b), (float("nan"), float("nan")))
                cells.append(f"{m:.3f} ± {sd:.3f}")
            lines.append(f"| {r} | " + " | ".join(cells) + " |")
        best = ", ".join(
            f"{b} steps -> R={max(LOOPS, key=lambda r: grid.get((r, b), (-1, 0))[0])}"
            for b in BUDGETS)
        lines += ["", f"Best depth per budget: {best}.", ""]

        lines += [f"### {h}-hop: unrolled past the trained depth", "",
                  "| trained R | budget | " +
                  " | ".join(f"eval R={e}" for e in (1, 2, 4, 8)) + " |",
                  "| --- | --- | " + " | ".join("---" for _ in range(4)) + " |"]
        for r in LOOPS:
            for b in BUDGETS:
                runs = load_results(f"e1_hops{h}_R{r}_b{b}_s")
                if not runs:
                    continue
                cells = []
                for e in (1, 2, 4, 8):
                    vals = [x["by_depth"][str(e)]["acc"] for x in runs.values()
                            if str(e) in x.get("by_depth", {})]
                    cells.append(f"{mean_sd(vals)[0]:.3f}" if vals else "-")
                lines.append(f"| {r} | {b} | " + " | ".join(cells) + " |")
        lines.append("")

        lines += [f"### {h}-hop: fixed compute (equal loops x steps)", "",
                  "| loops | steps | train FLOPs | accuracy |",
                  "| --- | --- | --- | --- |"]
        for (r, b) in ((1, 3000), (2, 1500)):
            got = load_results(f"e1_hops{h}_R{r}_b{b}_s")
            if not got:
                continue
            m, sd = mean_sd([x["final"]["acc"] for x in got.values()])
            flops = list(got.values())[0]["train_flops"]
            lines.append(f"| {r} | {b} | {flops:.2e} | {m:.3f} ± {sd:.3f} |")
        lines.append("")

    bgrid = _grid("e1_bytes", BYTE_LOOPS, BYTE_BUDGETS, "bpb")
    if bgrid:
        lines += ["## bytes: bits per byte (lower is better)", "",
                  "| loops | " + " | ".join(f"{b} steps" for b in BYTE_BUDGETS) + " |",
                  "| --- | " + " | ".join("---" for _ in BYTE_BUDGETS) + " |"]
        for r in BYTE_LOOPS:
            cells = []
            for b in BYTE_BUDGETS:
                m, sd = bgrid.get((r, b), (float("nan"), float("nan")))
                cells.append(f"{m:.4f} ± {sd:.4f}")
            lines.append(f"| {r} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--only", default="", help="'hops' or 'bytes' to run one half")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    if not args.report_only:
        selected = [j for j in jobs(seeds)
                    if not args.only or args.only in j["name"]]
        run_jobs(selected, workers=args.workers, dry_run=args.dry_run)
    if not args.dry_run:
        print(report())


if __name__ == "__main__":
    main()
