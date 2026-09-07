#!/usr/bin/env python3
"""A1: sweep the attention temperature, and watch the order parameter.

A head computes ``softmax(beta * q.k)``. That is a Boltzmann distribution over
keys with energy ``-q.k`` at inverse temperature ``beta``, and in a standard
transformer ``beta = 1/sqrt(head_dim)`` -- a variance-normalising constant that
nobody chose as a temperature. The theory collected in `../CALM/CRITICALITY.md`
§1 says this parameter has a phase diagram: **beta too low** (hot) and attention
is uniform and the head outputs an average (disorder, rank collapse); **beta too
high** (cold) and it becomes a hard argmax that copies one token (frozen); the
useful regime is in between, and the critical scale grows like ``log n``.

Nothing in this repo has measured where our model sits. This sweeps
``beta_scale`` -- a plain multiplier on ``1/sqrt(head_dim)`` -- at two depths,
and records normalised attention entropy per loop in every run.

Why do it at two depths: a weight-shared core runs the *same head* on a state
that our T2b measurement shows is contracting (state move 0.51 -> 0.19 -> 0.08).
If the logits sharpen as the state settles, entropy should fall across loops,
and depth saturation would be a head walking out of the useful phase rather
than only a state reaching a fixed point. Those have different fixes, which is
why the distinction is worth a run.

Registered predictions:

- **A1a.** Accuracy against ``beta_scale`` has an interior optimum. If accuracy
  is flat over a 16x range of temperature, this model is nowhere near either
  phase boundary and the whole criticality framing is inapplicable at n=37.
- **A1b.** Entropy falls monotonically across loops at R=4.
- **A1c.** The optimum sits at ``beta_scale > 1`` -- a *higher* beta, i.e.
  sharper and colder attention. Basis: entropy measured at initialisation is
  0.82-0.89 of maximum, which is the disordered (hot) side, so the useful
  direction looks like sharpening. This is the prediction most likely to be
  wrong and the cheapest to check.

  (Note on wording: beta is the *inverse* temperature, so high beta is cold and
  sharp, low beta is hot and uniform. Earlier drafts of this suite used
  "colder" for lower beta, which is backwards; the arm named ``cold`` in
  ``a3_cold_plus_router.py`` is in fact the *hot*, soft-attention arm. The name
  is kept because result files carry it.)

Usage::

    python a1_temperature.py --seeds 0,1 --workers 4
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
SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)
LOOPS = (1, 4)


def _tag(scale: float) -> str:
    return f"{scale:g}".replace(".", "p")


def jobs(seeds, steps: int = STEPS):
    return [{"name": f"a1_R{r}_b{_tag(sc)}_s{seed}", "task": "hops",
             "spec": hop_spec(HOP),
             "cfg": {**HOPS_CFG, "loops": r, "beta_scale": sc},
             "steps": steps, "batch": BATCH, "lr": LR, "seed": seed}
            for r in LOOPS for sc in SCALES for seed in seeds]


def report() -> str:
    lines = ["# A1 -- attention temperature sweep", "",
             "## accuracy against beta (x 1/sqrt(head_dim))", "",
             "| loops | " + " | ".join(f"x{sc:g}" for sc in SCALES) + " | best |",
             "| --- | " + " | ".join("---" for _ in SCALES) + " | --- |"]
    for r in LOOPS:
        cells, means = [], {}
        for sc in SCALES:
            runs = load_results(f"a1_R{r}_b{_tag(sc)}_s")
            if not runs:
                cells.append("-")
                continue
            m, sd = mean_sd([x["final"]["acc"] for x in runs.values()])
            means[sc] = m
            cells.append(f"{m:.3f} ± {sd:.3f}")
        best = f"x{max(means, key=means.get):g}" if means else "-"
        lines.append(f"| {r} | " + " | ".join(cells) + f" | **{best}** |")
    lines.append("")

    lines += ["## the order parameter: normalised attention entropy", "",
              "1.0 = uniform attention (disordered); 0.0 = hard argmax (frozen).",
              "",
              "| loops | beta | entropy per loop | final |",
              "| --- | --- | --- | --- |"]
    for r in LOOPS:
        for sc in SCALES:
            runs = load_results(f"a1_R{r}_b{_tag(sc)}_s")
            if not runs:
                continue
            first = list(runs.values())[0]
            ent = first.get("attention_entropy") or []
            if not ent:
                continue
            series = ", ".join(f"{e:.3f}" for e in ent)
            lines.append(f"| {r} | x{sc:g} | {series} | {ent[-1]:.3f} |")
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
