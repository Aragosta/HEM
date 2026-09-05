#!/usr/bin/env python3
"""E0: does the K3 baseline work at this size, and what is the noise floor?

Nothing later can be read without the two numbers E0 measures.

**1. The seed spread.** Several seeds of the *same* config on each task. Any
later difference smaller than this is reported as inside noise. Skipping this
is the commonest way a small-scale ablation study reports its own variance as a
finding.

**2. Whether the two K3 components survive the scale-down.** The arms are a
2x2 -- Attention Residuals on/off crossed with MoE/dense -- rather than two
one-at-a-time ablations, because the two mechanisms could easily interact
(AttnRes changes what each sublayer reads; the MoE changes what each sublayer
computes) and a factorial costs one extra cell to find that out.

This is not a formality. The pilot ladder in `RESULTS.md` found that Full
AttnRes, implemented from the report's equations, is a *drag* at this scale
large enough to decide the design of every later experiment: the arm without it
solves the composition task in about a thousand steps, and the arm with it is
still at chance. E0 is where that is measured properly, with seeds, on both
task families, so that the decision to run E1-E4 on standard residuals rests on
a measurement rather than on a pilot.

That result is about *this* scale and should not be read as a claim about
AttnRes: the report's own setting is 32+ layers and a compute budget six orders
of magnitude larger, and its mechanism (mitigating PreNorm dilution over depth)
is one that eight sublayers cannot exhibit.

Usage::

    python e0_baseline.py --hop-seeds 0,1,2 --byte-seeds 0,1 --workers 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import load_results, mean_sd, run_jobs                # noqa: E402
from spec import (BATCH, BYTE_BATCH, BYTE_CFG, BYTE_LR, BYTE_SEQ, BYTE_STEPS,
                  HOP_SPEC, HOPS_CFG, LR, STEPS)                  # noqa: E402

ARMS = {                                   # AttnRes x FFN, fully crossed
    "k3": {},                                             # AttnRes + MoE
    "noattnres": {"attn_res": "none"},                    # -       + MoE
    "dense": {"moe": False},                              # AttnRes + dense
    "noattnres_dense": {"attn_res": "none", "moe": False},
}


def jobs(hop_seeds, byte_seeds, steps: int, byte_steps: int):
    out = []
    for arm, delta in ARMS.items():
        for seed in hop_seeds:
            out.append({"name": f"e0_hops_{arm}_s{seed}", "task": "hops",
                        "spec": HOP_SPEC, "cfg": {**HOPS_CFG, "loops": 1, **delta},
                        "steps": steps, "batch": BATCH, "lr": LR, "seed": seed,
                        "eval_every": steps // 6})
        for seed in byte_seeds:
            out.append({"name": f"e0_bytes_{arm}_s{seed}", "task": "bytes:wikitext2",
                        "cfg": {**BYTE_CFG, "loops": 1, **delta},
                        "steps": byte_steps, "batch": BYTE_BATCH,
                        "seq_len": BYTE_SEQ, "lr": BYTE_LR, "seed": seed})
    return out


def report() -> str:
    lines = ["# E0 -- baseline and noise floor", ""]
    for task, metric, better in (("hops", "acc", "higher is better"),
                                 ("bytes", "bpb", "lower is better")):
        lines += [f"## {task} -- {metric} ({better})", "",
                  "| arm | AttnRes | FFN | mean | sd | n | s/run |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for arm in ARMS:
            runs = load_results(f"e0_{task}_{arm}_s")
            if not runs:
                continue
            values = [r["final"][metric] for r in runs.values()]
            secs = [r["seconds"] for r in runs.values()]
            m, sd = mean_sd(values)
            attn = "full" if "noattnres" not in arm else "none"
            ffn = "dense" if "dense" in arm else "MoE"
            lines.append(f"| {arm} | {attn} | {ffn} | {m:.4f} | {sd:.4f} | "
                         f"{len(values)} | {sum(secs) / len(secs):.0f} |")
        lines.append("")

        cells = {}
        for arm in ARMS:
            runs = load_results(f"e0_{task}_{arm}_s")
            if runs:
                cells[arm] = mean_sd([r["final"][metric] for r in runs.values()])[0]
        if len(cells) == 4:
            attn_moe = cells["k3"] - cells["noattnres"]
            attn_dense = cells["dense"] - cells["noattnres_dense"]
            lines += [
                f"AttnRes effect with MoE: {attn_moe:+.4f}; "
                f"with a dense FFN: {attn_dense:+.4f}; "
                f"interaction: {attn_moe - attn_dense:+.4f}.", ""]

        base = load_results(f"e0_{task}_k3_s")
        if base:
            values = [r["final"][metric] for r in base.values()]
            m, sd = mean_sd(values)
            lines += [f"**Noise floor ({task}): sd {sd:.4f} over {len(values)} "
                      f"seeds of the K3 arm, mean {m:.4f}.**", ""]
        alt = load_results(f"e0_{task}_noattnres_s")
        if alt:
            values = [r["final"][metric] for r in alt.values()]
            m, sd = mean_sd(values)
            lines += [f"**Noise floor ({task}, standard residuals -- the arm E1-E4 "
                      f"are run on): sd {sd:.4f} over {len(values)} seeds, "
                      f"mean {m:.4f}.**", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hop-seeds", default="0,1,2")
    ap.add_argument("--byte-seeds", default="0,1")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--byte-steps", type=int, default=BYTE_STEPS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    hop_seeds = [int(s) for s in args.hop_seeds.split(",")]
    byte_seeds = [int(s) for s in args.byte_seeds.split(",")]
    if not args.report_only:
        selected = [j for j in jobs(hop_seeds, byte_seeds, args.steps,
                                    args.byte_steps)
                    if not args.only or args.only in j["name"]]
        run_jobs(selected, workers=args.workers, dry_run=args.dry_run)
    if not args.dry_run:
        print(report())


if __name__ == "__main__":
    main()
