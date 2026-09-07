#!/usr/bin/env python3
"""A4: does the optimal temperature scale like 1/d or like 1/sqrt(d)?

A1 found that the best attention temperature at R=4 is **x0.25 of the default**.
The default is `1/sqrt(head_dim)` and our head_dim is 16, so the default is
`1/4 = 0.25` and the empirical optimum is `0.0625 = 1/16 = 1/head_dim` --
*exactly* the scaling that maximal update parametrization (muP, Tensor Programs
V, arXiv:2203.03466) prescribes for attention logits: **1/d, not 1/sqrt(d)**.

muP's stated reason is the same mechanism A1's failed prediction ran into. The
`1/sqrt(d)` constant is derived assuming q and k are independent, which is true
at initialisation and false afterwards: training makes them correlated, the
logits grow like `d` rather than `sqrt(d)`, and the standard scaling
under-corrects. That is why our trained entropy sits at 0.41-0.53 while the
untrained model measures 0.82-0.89, and why the useful correction is *colder*.

If that reading is right, the coincidence is not a coincidence and it must
reproduce at other head dimensions. The test is a sweep of head_dim against the
temperature multiplier, holding `dim` (and therefore the parameter count)
fixed:

| head_dim | default beta | muP prediction (1/d) | multiplier that would give it |
| --- | --- | --- | --- |
| 32 | 0.177 | 0.031 | **x0.177** |
| 16 | 0.250 | 0.062 | **x0.250** |
| 8 | 0.354 | 0.125 | **x0.354** |

Registered predictions:

- **A4a (muP).** The optimal multiplier *falls* as head_dim rises, tracking
  `1/sqrt(head_dim)` -- 0.177 / 0.25 / 0.354 for head_dim 32 / 16 / 8.
- **A4b (the null).** The optimal multiplier is roughly constant across
  head_dim. That would mean `1/sqrt(d)` is already the right functional form and
  A1's optimum was a property of this task, not of the parametrisation.
- **A4c.** Whichever holds, attention entropy at the optimum should be similar
  across head_dim -- the temperature is doing the same job in each case, and
  entropy is the quantity being held fixed.

A4a and A4b make opposite predictions about the *direction* of a trend across
three points, which is about as much as two seeds per cell can support. This is
a check on a hypothesis, not a scaling law.

Usage::

    python a4_head_dim.py --seeds 0,1 --workers 4
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
HEADS = (2, 4, 8)                 # head_dim = 64 / n_heads = 32, 16, 8
SCALES = (0.125, 0.25, 0.5, 1.0)


def _tag(scale: float) -> str:
    return f"{scale:g}".replace(".", "p")


def jobs(seeds, steps: int = STEPS):
    return [{"name": f"a4_h{h}_b{_tag(sc)}_s{seed}", "task": "hops",
             "spec": hop_spec(HOP),
             "cfg": {**HOPS_CFG, "n_heads": h, "loops": 4, "beta_scale": sc},
             "steps": steps, "batch": BATCH, "lr": LR, "seed": seed}
            for h in HEADS for sc in SCALES for seed in seeds]


def report() -> str:
    dim = HOPS_CFG["dim"]
    lines = ["# A4 -- optimal temperature against head dimension", "",
             "| head_dim | " + " | ".join(f"x{sc:g}" for sc in SCALES) +
             " | best | 1/sqrt(d) predicts |",
             "| --- | " + " | ".join("---" for _ in SCALES) + " | --- | --- |"]
    for h in HEADS:
        head_dim = dim // h
        cells, means = [], {}
        for sc in SCALES:
            runs = load_results(f"a4_h{h}_b{_tag(sc)}_s")
            if not runs:
                cells.append("-")
                continue
            m, sd = mean_sd([x["final"]["acc"] for x in runs.values()])
            means[sc] = m
            cells.append(f"{m:.3f} ± {sd:.3f}")
        best = f"x{max(means, key=means.get):g}" if means else "-"
        lines.append(f"| {head_dim} | " + " | ".join(cells) +
                     f" | **{best}** | x{head_dim ** -0.5:.3f} |")
    lines.append("")

    lines += ["## absolute beta at the optimum, against the two candidate laws", "",
              "| head_dim | best absolute beta | 1/d (muP) | 1/sqrt(d) (standard) |",
              "| --- | --- | --- | --- |"]
    for h in HEADS:
        head_dim = dim // h
        means = {}
        for sc in SCALES:
            runs = load_results(f"a4_h{h}_b{_tag(sc)}_s")
            if runs:
                means[sc] = mean_sd([x["final"]["acc"] for x in runs.values()])[0]
        if not means:
            continue
        best = max(means, key=means.get)
        lines.append(f"| {head_dim} | {best * head_dim ** -0.5:.4f} | "
                     f"{1 / head_dim:.4f} | {head_dim ** -0.5:.4f} |")
    lines.append("")

    lines += ["## entropy at each cell (A4c)", "",
              "| head_dim | beta | final-loop entropy |", "| --- | --- | --- |"]
    for h in HEADS:
        for sc in SCALES:
            runs = load_results(f"a4_h{h}_b{_tag(sc)}_s")
            if not runs:
                continue
            ent = [r.get("attention_entropy") for r in runs.values()]
            ent = [e[-1] for e in ent if e]
            if ent:
                lines.append(f"| {dim // h} | x{sc:g} | "
                             f"{sum(ent) / len(ent):.3f} |")
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
