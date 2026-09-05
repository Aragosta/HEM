#!/usr/bin/env python3
"""E4: the halting gate, and whether the brief's fix to it buys anything.

Ouro's gate is a categorical distribution over exactly T=4 steps, trained with
an entropy regulariser against a **uniform prior**. The brief's §5 argues that
neither part survives unbounded depth: there is no uniform prior over an
unbounded set, and a head conditioned on the step index cannot extrapolate past
the indices it saw. The proposed fix is a **step-invariant** gate -- halting
probability as a function of the latent state alone -- with a **geometric**
prior, which is PonderNet.

That argument is cheap to test and this is the only part of the brief's halting
section that does not need RL. Three arms, trained at up to four loops:

| arm | gate | prior |
| --- | --- | --- |
| `none` | no gate; fixed depth | - |
| `ouro` | one logit per step index | uniform over T=4 |
| `pondernet` | one logit, same head at every step | geometric (lambda=0.5) |

and each is scored twice: at the trained depth, and **unrolled to 8 and 12**,
which is the only place the two formulations are predicted to differ. The
`none` arm is scored with Huginn's zero-shot KL exit rule -- exit when the
next-token distribution stops moving -- because it costs no training at all and
is therefore the reference any learned gate has to beat.

What the table reports is not accuracy but the *pair* (accuracy, average depth
spent). A gate that is more accurate because it always runs to the last step
has not done anything; the Q-exit rule makes the trade explicit.

Usage::

    python e4_halting.py --seeds 0,1,2 --workers 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import load_results, mean_sd, run_jobs                # noqa: E402
from spec import BATCH, HOP_SPEC, HOPS_CFG, LR, STEPS            # noqa: E402

ARMS = {
    "none": {},
    "ouro": {"halting": "ouro", "max_train_loops": 4},
    "pondernet": {"halting": "pondernet", "halt_prior": 0.5, "max_train_loops": 4},
}
EVAL_LOOPS = (1, 2, 4, 8, 12)


def jobs(seeds, steps: int):
    out = []
    for arm, delta in ARMS.items():
        for seed in seeds:
            out.append({"name": f"e4_{arm}_s{seed}", "task": "hops",
                        "spec": HOP_SPEC,
                        "cfg": {**HOPS_CFG, "loops": 4, **delta},
                        "steps": steps, "batch": BATCH, "lr": LR,
                        "seed": seed, "eval_loops": EVAL_LOOPS})
    return out


def report() -> str:
    lines = ["# E4 -- halting", "",
             "## accuracy at the trained depth and unrolled beyond it", "",
             "| arm | " + " | ".join(f"R={r}" for r in EVAL_LOOPS) + " |",
             "| --- | " + " | ".join("---" for _ in EVAL_LOOPS) + " |"]
    for arm in ARMS:
        runs = load_results(f"e4_{arm}_s")
        if not runs:
            continue
        cells = []
        for r in EVAL_LOOPS:
            vals = [x["by_depth"][str(r)]["acc"] for x in runs.values()
                    if str(r) in x.get("by_depth", {})]
            m, sd = mean_sd(vals)
            cells.append(f"{m:.3f} ± {sd:.3f}" if vals else "-")
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines.append("")

    lines += ["## what the exit rule spends", "",
              "| arm | eval depth | rule | accuracy | mean depth used |",
              "| --- | --- | --- | --- | --- |"]
    for arm in ARMS:
        runs = load_results(f"e4_{arm}_s")
        for name, r in sorted(runs.items()):
            for depth, block in sorted(r.get("exits", {}).items(), key=lambda kv: int(kv[0])):
                accs = {k: v for k, v in block.items() if "_acc_" in k}
                depths = {k: v for k, v in block.items() if "_depth_" in k}
                if not accs:
                    continue
                rules = sorted({k.split("_acc_")[0] for k in accs})
                for rule in rules:
                    a = [v for k, v in accs.items() if k.startswith(rule + "_acc")]
                    d = [v for k, v in depths.items() if k.startswith(rule + "_depth")]
                    lines.append(f"| {arm} ({name.rsplit('_', 1)[-1]}) | {depth} | "
                                 f"{rule} | {sum(a) / len(a):.3f} | "
                                 f"{sum(d) / len(d):.2f} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=2000)
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
