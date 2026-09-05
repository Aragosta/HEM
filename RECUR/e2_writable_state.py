#!/usr/bin/env python3
"""E2: does the loop need somewhere to write?

The brief's §7.3, and the highest-information experiment on its list. Chain of
thought buys *space* -- an unbounded external tape. Looping buys *depth* --
more sequential steps over a state of fixed width. If the binding constraint on
a looped model is state capacity rather than sequential compute, then depth
saturates because the scratchpad stopped fitting, and both papers' saturation
curves have an explanation neither offered.

Four arms at R=4, differing in one field each:

| arm | field | what it adds |
| --- | --- | --- |
| `plain` | - | the loop rewrites one state, as in both papers |
| `regs` | `registers=8` | eight writable positions carried across loops |
| `regswiped` | `register_persist=False` | the same eight positions, reset every loop |
| `xloop` | `loop_memory="attn_res"` | the loop reads every earlier loop's sublayer outputs |

`regswiped` is the control that makes the experiment worth running: it has the
same width, the same parameters and the same FLOPs as `regs` and differs only
in whether what was written survives the loop boundary. Without it, a win for
`regs` is indistinguishable from "the model got wider".

`xloop` is the K3 baseline's own mechanism turned on across the loop boundary
(`BASELINE.md` §"The one place K3 and the brief meet"): Attention Residuals
already give a layer content-addressed read access to every earlier layer's
output, and under a loop those earlier layers are earlier iterations. It costs
one vector per sublayer, against `registers`' extra sequence positions -- and
because it is a *read* mechanism rather than a *write* mechanism, if it matches
the register bank then what the loop needed was access, not storage.

The task carries the hypothesis: `twochain` asks for a function of two
independent chains, so one partial result has to survive while the other is
computed. `hops` is the control, where nothing needs holding. A register bank
that helps equally on both is not helping with state.

Usage::

    python e2_writable_state.py --seeds 0,1,2 --workers 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import load_results, mean_sd, run_jobs                # noqa: E402
from spec import BATCH, HOP_SPEC, HOPS_CFG, LR                    # noqa: E402

ARMS = {
    "plain": {},
    "regs": {"registers": 8},
    "regswiped": {"registers": 8, "register_persist": False},
    "xloop": {"loop_memory": "attn_res"},
}
TASKS = ("twochain", "hops")


def jobs(seeds, steps: int):
    out = []
    for task in TASKS:
        for arm, delta in ARMS.items():
            for seed in seeds:
                out.append({"name": f"e2_{task}_{arm}_s{seed}", "task": task,
                            "spec": HOP_SPEC,
                            "cfg": {**HOPS_CFG, "loops": 4, **delta},
                            "steps": steps, "batch": BATCH, "lr": LR,
                            "seed": seed})
    return out


def report() -> str:
    lines = ["# E2 -- where the loop may write", ""]
    for task in TASKS:
        lines += [f"## {task} (accuracy)", "",
                  "| arm | mean | sd | n | vs plain | params | FLOPs/token |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        base = None
        for arm in ARMS:
            runs = load_results(f"e2_{task}_{arm}_s")
            if not runs:
                continue
            values = [r["final"]["acc"] for r in runs.values()]
            m, sd = mean_sd(values)
            if arm == "plain":
                base = m
            one = list(runs.values())[0]
            delta = "-" if base is None or arm == "plain" else f"{m - base:+.3f}"
            lines.append(f"| {arm} | {m:.3f} | {sd:.3f} | {len(values)} | {delta} | "
                         f"{one['params']:,} | {one['flops_per_token']:.2e} |")
        lines.append("")

        # paired differences, which is how these should be read
        lines += ["Paired per-seed differences against `plain`:", ""]
        plain = load_results(f"e2_{task}_plain_s")
        for arm in ARMS:
            if arm == "plain":
                continue
            runs = load_results(f"e2_{task}_{arm}_s")
            diffs = []
            for name, r in runs.items():
                seed = name.rsplit("_s", 1)[-1]
                mate = plain.get(f"e2_{task}_plain_s{seed}")
                if mate:
                    diffs.append(r["final"]["acc"] - mate["final"]["acc"])
            if diffs:
                m, sd = mean_sd(diffs)
                lines.append(f"- `{arm}`: " +
                             ", ".join(f"{d:+.3f}" for d in diffs) +
                             f" (mean {m:+.3f}, sd {sd:.3f})")
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
