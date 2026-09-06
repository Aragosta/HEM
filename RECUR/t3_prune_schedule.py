#!/usr/bin/env python3
"""T3: overproduce then prune -- and does the loop want it only on the first pass?

Developmental neuroscience's version of sparsity is not "train sparse". Synapses
are massively over-produced and then eliminated, fast at first and slowly after
(*Decreasing-Rate Pruning Optimizes the Construction of Efficient and Robust
Distributed Networks*, PLOS Comp Biol pcbi.1004347), and the resulting networks
are both more efficient and more robust than networks grown sparse. A companion
result (pcbi.1009458) shows purely **local** synaptic rules solve the global
architecture problem -- which is what T4 tests.

Here we test the schedule itself, plus one thing neither literature can ask
because neither has a loop. Our E3 result said the loop wants a **phase
distinction**: the first pass differs from the refinement passes, and the cheap
conditioner that provides exactly that beat the expressive one. If that is
right, the developmental transient may be needed *per loop position* rather than
per training step -- wide entry, narrow refinement, permanently.

Four arms at R=2 on the 3-hop task, differing in one field:

| arm | field | what it says |
| --- | --- | --- |
| `k2` | `n_active=2` | grow sparse and stay sparse (the E0-E4 default) |
| `k8` | `n_active=8` | stay dense; the cost ceiling |
| `anneal` | `k_anneal=[8,2]` | overproduce then prune, decreasing-rate |
| `phase` | `k_by_loop=(8,2)` | wide entry, narrow refinement -- forever |

Registered predictions:

- **T3a.** `anneal` >= `k2` at equal final sparsity. If the dense transient does
  nothing, the developmental result does not transfer to a transformer's expert
  graph and that is worth knowing cheaply.
- **T3b.** `phase` >= `anneal`. This is the one that would be new: it says the
  transient is about *position in the loop*, not position in training, and it
  follows from E3 rather than from either literature.
- **T3c.** `k8` is not the best arm. If simply activating more experts wins,
  T1's k* is high and this experiment is measuring the wrong knob.

Usage::

    python t3_prune_schedule.py --seeds 0,1 --workers 4
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
    "k2": {"cfg": {"n_active": 2}},
    "k8": {"cfg": {"n_active": 8}},
    "anneal": {"cfg": {"n_active": 2}, "k_anneal": [8, 2]},
    "phase": {"cfg": {"n_active": 2, "k_by_loop": (8, 2)}},
}


def jobs(seeds, steps: int = STEPS):
    out = []
    for arm, delta in ARMS.items():
        for seed in seeds:
            job = {"name": f"t3_{arm}_s{seed}", "task": "hops",
                   "spec": hop_spec(HOP),
                   "cfg": {**HOPS_CFG, "loops": 2, **delta["cfg"]},
                   "steps": steps, "batch": BATCH, "lr": LR, "seed": seed}
            if "k_anneal" in delta:
                job["k_anneal"] = delta["k_anneal"]
            out.append(job)
    return out


def report() -> str:
    lines = ["# T3 -- overproduce-then-prune, and the per-loop version", "",
             "| arm | accuracy | sd | n | mean experts/token at the end | FLOPs/token |",
             "| --- | --- | --- | --- | --- | --- |"]
    for arm in ARMS:
        runs = load_results(f"t3_{arm}_s")
        if not runs:
            continue
        m, sd = mean_sd([x["final"]["acc"] for x in runs.values()])
        ks = []
        for r in runs.values():
            state = r.get("routing_state", {})
            if state.get("mean_k"):
                ks.append(sum(state["mean_k"]) / len(state["mean_k"]))
        one = list(runs.values())[0]
        k_txt = f"{sum(ks) / len(ks):.2f}" if ks else "-"
        lines.append(f"| {arm} | {m:.3f} | {sd:.3f} | {len(runs)} | {k_txt} | "
                     f"{one['flops_per_token']:.2e} |")
    lines.append("")

    base = load_results("t3_k2_s")
    if base:
        lines += ["Paired per-seed differences against `k2` (same seed, same "
                  "data stream, same weights where shared):", ""]
        for arm in ARMS:
            if arm == "k2":
                continue
            runs = load_results(f"t3_{arm}_s")
            diffs = []
            for name, r in runs.items():
                mate = base.get(f"t3_k2_s{name[-1]}")
                if mate:
                    diffs.append(r["final"]["acc"] - mate["final"]["acc"])
            if diffs:
                m, sd = mean_sd(diffs)
                lines.append(f"- `{arm}`: " + ", ".join(f"{d:+.3f}" for d in diffs)
                             + f" (mean {m:+.3f})")
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
