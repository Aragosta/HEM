#!/usr/bin/env python3
"""A5: train soft, infer sharp? Two literatures point opposite ways.

**Ours (A1).** Sweeping the training temperature, a *lower* beta wins:
0.678 / 0.639 / 0.608 / 0.607 / 0.456 at x0.25 / x0.5 / x1 / x2 / x4, R=4. Soft
attention trains better, and the mechanism is that sharp attention kills its own
gradient -- the softmax Jacobian `diag(a) - aa^T` vanishes when one weight is
~1, so a head that commits early cannot learn that it committed wrongly.

**Velickovic et al., arXiv:2410.01104.** Softmax provably *disperses* as the
number of items grows, so a head that must pick a maximum gets blurrier out of
distribution. Their fix is **adaptive temperature at inference only** -- sharpen
the coefficients after training, with no change to the weights.

Both can be true, and if they are, the schedule is **soft while learning, sharp
while deciding**. That is checkable for free: beta is a multiplier on the
logits, so a trained model can be evaluated at any temperature without
retraining. `harness.beta_transfer` does exactly that, and this experiment
trains at three temperatures and evaluates each model at five.

Registered predictions:

- **A5a.** The best *evaluation* temperature is higher (sharper) than the
  training temperature the model was trained at. This is the synthesis of the
  two results above and the reason to run it.
- **A5b.** A model trained soft (x0.25) and evaluated sharp beats a model
  trained *and* evaluated at the default. If it does, "train soft, infer sharp"
  is a free improvement over the standard recipe, needing no architecture
  change at all.
- **A5c, the honest null.** Transfer is flat: accuracy depends on the training
  temperature and not on the evaluation temperature. That would mean the model
  has already absorbed the temperature into ||q|| and ||k|| during training,
  which is itself worth knowing -- it would say the knob only matters while
  gradients flow, and that inference-time sharpening (their result) does not
  reproduce on a task where the circuit was learned in distribution.

Usage::

    python a5_train_soft_infer_sharp.py --seeds 0,1 --workers 3
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
TRAIN_SCALES = (0.25, 1.0, 4.0)
EVAL_SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)


def _tag(scale: float) -> str:
    return f"{scale:g}".replace(".", "p")


def jobs(seeds, steps: int = STEPS):
    return [{"name": f"a5_train{_tag(sc)}_s{seed}", "task": "hops",
             "spec": hop_spec(HOP),
             "cfg": {**HOPS_CFG, "loops": 4, "beta_scale": sc},
             "steps": steps, "batch": BATCH, "lr": LR, "seed": seed}
            for sc in TRAIN_SCALES for seed in seeds]


def report() -> str:
    lines = ["# A5 -- training temperature against evaluation temperature", "",
             "Rows are what the model was trained at, columns what it was "
             "evaluated at. The diagonal is the usual number; everything else "
             "costs nothing.", "",
             "| trained at | " + " | ".join(f"eval x{s:g}" for s in EVAL_SCALES) +
             " | best eval |",
             "| --- | " + " | ".join("---" for _ in EVAL_SCALES) + " | --- |"]
    for tsc in TRAIN_SCALES:
        runs = load_results(f"a5_train{_tag(tsc)}_s")
        if not runs:
            continue
        cells, means = [], {}
        for esc in EVAL_SCALES:
            key = f"eval_beta_x{esc:g}"
            vals = [r["beta_transfer"][key] for r in runs.values()
                    if key in r.get("beta_transfer", {})]
            if not vals:
                cells.append("-")
                continue
            m, sd = mean_sd(vals)
            means[esc] = m
            mark = "**" if abs(esc - tsc) < 1e-9 else ""
            cells.append(f"{mark}{m:.3f}{mark}")
        best = f"x{max(means, key=means.get):g}" if means else "-"
        lines.append(f"| x{tsc:g} | " + " | ".join(cells) + f" | {best} |")
    lines.append("")

    best_overall = None
    for tsc in TRAIN_SCALES:
        runs = load_results(f"a5_train{_tag(tsc)}_s")
        for esc in EVAL_SCALES:
            key = f"eval_beta_x{esc:g}"
            vals = [r["beta_transfer"][key] for r in runs.values()
                    if key in r.get("beta_transfer", {})]
            if vals:
                m = mean_sd(vals)[0]
                if best_overall is None or m > best_overall[0]:
                    best_overall = (m, tsc, esc)
    if best_overall:
        m, tsc, esc = best_overall
        lines += [f"Best cell: train x{tsc:g}, evaluate x{esc:g} -> {m:.3f}.", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--workers", type=int, default=3)
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
