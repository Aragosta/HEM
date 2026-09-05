#!/usr/bin/env python3
"""E3: can a weight-tied loop compute different things at different steps?

The standing objection to weight tying is that every iteration applies the same
function. The brief's §7.2 answer is that a MoE core need not: condition the
*router* on the loop index and iteration 1 can activate different experts from
iteration 4, giving functional differentiation for the price of a small table.
Since the baseline is already an MoE, this costs almost nothing to test.

Three arms at R=4, differing in one field:

| arm | `step_routing` | what the loop index can change |
| --- | --- | --- |
| `none` | `"none"` | nothing: identical routing function every loop |
| `bias` | `"bias"` | a per-loop bias on the expert logits -- can reorder experts globally, cannot make the ordering token-dependent |
| `embed` | `"embed"` | a per-loop vector added to the router input -- can change *which* tokens go where |

`bias` is the control that separates "the loop needs different experts" from
"the loop needs a different *assignment* of tokens to experts". Both are
cheap; only the second is the mechanism the brief argues for.

Two things are measured, and the second is the more informative:

1. **accuracy**, on both task families;
2. **whether routing actually diverges across loops** -- the Jensen-Shannon
   divergence between the expert-usage distributions of loop 1 and loop R, and
   the drift in expert load. A conditioner that changes nothing has failed even
   if accuracy moves, and a conditioner that differentiates routing without
   moving accuracy is a real (and reportable) null: the expressivity was
   available and unused.

The load statistics also address the brief's untested risk -- that MoE routing
instability and recurrent-depth instability compound. The aux-loss-free bias is
updated once per optimiser step regardless of R, so a looped model makes R
times as many routing decisions per update as a plain one. If that destabilises
load, load variance should grow with R, and E1's runs record the same statistic
at R = 1, 2, 4 and 8 for exactly that comparison.

Usage::

    python e3_step_routing.py --seeds 0,1,2 --workers 4
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import load_results, mean_sd, run_jobs                # noqa: E402
from spec import (BYTE_BATCH, BYTE_CFG, BYTE_LR, BYTE_SEQ, BYTE_STEPS,
                  BATCH, HOP_SPEC, HOPS_CFG, LR, STEPS)           # noqa: E402

ARMS = {"none": {}, "bias": {"step_routing": "bias"},
        "embed": {"step_routing": "embed"}}


def jobs(seeds, steps: int, byte_steps: int):
    out = []
    for arm, delta in ARMS.items():
        for seed in seeds:
            out.append({"name": f"e3_hops_{arm}_s{seed}", "task": "hops",
                        "spec": HOP_SPEC,
                        "cfg": {**HOPS_CFG, "loops": 4, **delta},
                        "steps": steps, "batch": BATCH, "lr": LR, "seed": seed})
            out.append({"name": f"e3_bytes_{arm}_s{seed}",
                        "task": "bytes:wikitext2",
                        "cfg": {**BYTE_CFG, "loops": 4, **delta},
                        "steps": byte_steps, "batch": BYTE_BATCH,
                        "seq_len": BYTE_SEQ, "lr": BYTE_LR, "seed": seed})
    return out


def js_divergence(p, q):
    p = [x / (sum(p) or 1) for x in p]
    q = [x / (sum(q) or 1) for x in q]
    m = [(a + b) / 2 for a, b in zip(p, q)]

    def kl(a, b):
        return sum(x * math.log(x / y) for x, y in zip(a, b) if x > 0 and y > 0)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def routing_divergence(result) -> float:
    """JS divergence between loop-1 and loop-R expert usage, averaged over blocks.

    ``routing`` in a collected forward is one index tensor per (loop, MoE
    block); the first ``n_core`` entries are loop 1 and the last ``n_core`` are
    loop R.
    """
    routing = result.get("routing_hist")
    if not routing:
        return float("nan")
    first, last = routing[0], routing[-1]
    return js_divergence(first, last)


def report() -> str:
    lines = ["# E3 -- loop-index-conditioned MoE routing", ""]
    for task, metric in (("hops", "acc"), ("bytes", "bpb")):
        lines += [f"## {task} ({metric})", "",
                  "| arm | mean | sd | n | load sd (last block) |",
                  "| --- | --- | --- | --- | --- |"]
        for arm in ARMS:
            runs = load_results(f"e3_{task}_{arm}_s")
            if not runs:
                continue
            values = [r["final"][metric] for r in runs.values()]
            m, sd = mean_sd(values)
            loads = []
            for r in runs.values():
                if r.get("expert_load"):
                    last = r["expert_load"][-1]
                    mean = sum(last) / len(last)
                    loads.append((sum((x - mean) ** 2 for x in last) / len(last)) ** 0.5)
            load = f"{sum(loads) / len(loads):.4f}" if loads else "-"
            lines.append(f"| {arm} | {m:.4f} | {sd:.4f} | {len(values)} | {load} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--byte-steps", type=int, default=BYTE_STEPS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    if not args.report_only:
        selected = [j for j in jobs(seeds, args.steps, args.byte_steps)
                    if not args.only or args.only in j["name"]]
        run_jobs(selected, workers=args.workers, dry_run=args.dry_run)
    if not args.dry_run:
        print(report())


if __name__ == "__main__":
    main()
