#!/usr/bin/env python3
"""A2: per-loop temperature against per-loop routing -- and do they add?

E3 and T2b found the same thing twice: a **per-loop bias on the expert router**
is the only intervention in this project that repeatedly beats its baseline
(+0.037 at R=4 on 2 hops, +0.063 at R=4 on 3 hops). The reading offered for it
was that a weight-tied loop wants a *phase distinction* -- the first pass should
differ from the refinement passes -- not per-step specialisation.

If that reading is right, the router is the wrong place to apply it. Routing is
a linear map of the hidden state (arXiv:2604.09780), and T2b measured that
routing convergence and state convergence are one curve (r = -0.877). The state
is written by attention. So the same symmetry-breaking applied *upstream*, at
the attention temperature, should do at least as much work -- and if the two are
the same intervention seen at two points in one pipeline, **their gains should
not add**.

Four arms at R=4, one field apart:

| arm | field | where the loop index enters |
| --- | --- | --- |
| `none` | - | nowhere |
| `beta` | `beta_mode="per_loop"` | attention temperature, per (loop, head) |
| `bias` | `step_routing="bias"` | expert-router logits, per loop |
| `both` | both | both |

Registered predictions:

- **A2a.** `beta` >= `none`. The temperature is the upstream knob.
- **A2b, the informative one.** `both` is close to `max(beta, bias)` rather than
  to their sum. Non-additivity is the evidence that the two interventions are
  the same one; additivity would mean they act on different things and the
  upstream/downstream story is wrong.
- **A2c.** `beta` costs 16 parameters per attention layer (4 heads x 4 loops)
  against `bias`'s 64 (16 experts x 4 loops), so if they tie, the temperature is
  the cheaper mechanism and the one to keep.

Usage::

    python a2_beta_vs_router.py --seeds 0,1 --workers 4
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
    "none": {},
    "beta": {"beta_mode": "per_loop"},
    "bias": {"step_routing": "bias"},
    "both": {"beta_mode": "per_loop", "step_routing": "bias"},
}


def jobs(seeds, steps: int = STEPS):
    return [{"name": f"a2_{arm}_s{seed}", "task": "hops", "spec": hop_spec(HOP),
             "cfg": {**HOPS_CFG, "loops": 4, **delta},
             "steps": steps, "batch": BATCH, "lr": LR, "seed": seed}
            for arm, delta in ARMS.items() for seed in seeds]


def report() -> str:
    lines = ["# A2 -- per-loop temperature vs per-loop routing", "",
             "| arm | accuracy | sd | n | vs none | params |",
             "| --- | --- | --- | --- | --- | --- |"]
    means = {}
    base = None
    for arm in ARMS:
        runs = load_results(f"a2_{arm}_s")
        if not runs:
            continue
        m, sd = mean_sd([r["final"]["acc"] for r in runs.values()])
        means[arm] = m
        if arm == "none":
            base = m
        delta = "-" if base is None or arm == "none" else f"{m - base:+.3f}"
        lines.append(f"| {arm} | {m:.3f} | {sd:.3f} | {len(runs)} | {delta} | "
                     f"{list(runs.values())[0]['params']:,} |")
    lines.append("")

    if {"none", "beta", "bias", "both"} <= set(means):
        d_beta = means["beta"] - means["none"]
        d_bias = means["bias"] - means["none"]
        d_both = means["both"] - means["none"]
        lines += ["## additivity (A2b)", "",
                  f"- temperature alone: {d_beta:+.3f}",
                  f"- routing alone: {d_bias:+.3f}",
                  f"- both together: {d_both:+.3f}",
                  f"- sum of the two singles: {d_beta + d_bias:+.3f}",
                  f"- larger single: {max(d_beta, d_bias):+.3f}",
                  "",
                  "Closer to the larger single than to the sum means the two "
                  "are one intervention applied at two points in the same "
                  "pipeline; closer to the sum means they are different "
                  "mechanisms and the upstream story is wrong.", ""]

    lines += ["## what the temperature learned, per loop", "",
              "| arm | seed | entropy per loop |", "| --- | --- | --- |"]
    for arm in ARMS:
        for name, r in sorted(load_results(f"a2_{arm}_s").items()):
            ent = r.get("attention_entropy") or []
            if ent:
                lines.append(f"| {arm} | {name[-1]} | " +
                             ", ".join(f"{e:.3f}" for e in ent) + " |")
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
