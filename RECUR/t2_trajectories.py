#!/usr/bin/env python3
"""T2: follow each token's expert path across loops. Analysis only -- no new runs.

E3 measured that routing changes sharply from loop 1 to loop 2 and then stops
(JS 0.047 -> 0.003 -> 0.0003). *The Myth of Expert Specialization in MoEs*
(arXiv:2604.09780) supplies the reason that matters: routers are linear maps, so
expert-usage similarity is explained by hidden-state similarity -- specialisation
is a property of the representation, not the routing architecture. Put those
together and routing convergence is a **cheap read-out of state convergence**.

This experiment tests that identification and then tries to get something useful
out of it. Every training run in T1/T3/T4 records, at the end:

* `overlap` -- Jaccard between a token's expert set at consecutive loops;
* `state_move` -- the relative size of the state update at the same loop;
* `settle` -- the last loop at which a token's expert set changed, reported
  separately for questions the model got right and wrong.

Registered predictions:

- **T2a.** `overlap` rises toward 1 and `state_move` falls toward 0 on the same
  schedule. They are two views of one contraction; if they diverge, the
  linear-router argument does not hold here and E3's convergence story needs
  another explanation.
- **T2b.** Tokens the model gets **wrong settle later** than tokens it gets
  right. That would make the expert path a difficulty signal.
- **T2c, the payoff if T2b holds.** A halting rule that exits when the expert
  set stops changing costs a set comparison, where Huginn's KL rule costs a
  softmax over the vocabulary -- and E4 established that the KL rule already
  matches the trained gates. A cheaper rule at the same accuracy would be the
  most useful artefact in this folder, and it is also *legible*: expert indices
  are discrete and nameable, which is exactly what latent reasoning is accused
  of lacking.

Usage::

    python t2_trajectories.py                # reads results/, prints tables
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import load_results, mean_sd                          # noqa: E402

FAMILIES = ("t1_", "t3_", "t4_")


def _rows():
    for family in FAMILIES:
        for name, run in sorted(load_results(family).items()):
            traj = run.get("trajectories")
            if traj:
                yield name, run, traj


def report() -> str:
    lines = ["# T2 -- expert trajectories across loops", ""]

    lines += ["## does routing convergence track state convergence? (T2a)", "",
              "| run | loop | expert-set overlap | relative state move |",
              "| --- | --- | --- | --- |"]
    shown = 0
    for name, run, traj in _rows():
        hops = run["spec"]["hops"][0]
        overlaps = traj.get(f"overlap_h{hops}") or []
        moves = traj.get(f"state_move_h{hops}") or []
        if not overlaps or shown >= 12:
            continue
        shown += 1
        for i, ov in enumerate(overlaps):
            mv = f"{moves[i]:.3f}" if i < len(moves) else "-"
            lines.append(f"| {name} | {i + 1}->{i + 2} | {ov:.3f} | {mv} |")
    lines.append("")

    lines += ["## do wrong answers settle later? (T2b)", "",
              "| family | mean settle step (correct) | (wrong) | difference | n runs |",
              "| --- | --- | --- | --- | --- |"]
    for family in FAMILIES:
        right, wrong = [], []
        for name, run, traj in _rows():
            if not name.startswith(family):
                continue
            hops = run["spec"]["hops"][0]
            if f"settle_correct_h{hops}" in traj and f"settle_wrong_h{hops}" in traj:
                right.append(traj[f"settle_correct_h{hops}"])
                wrong.append(traj[f"settle_wrong_h{hops}"])
        if not right:
            continue
        mr, _ = mean_sd(right)
        mw, _ = mean_sd(wrong)
        lines.append(f"| {family.rstrip('_')} | {mr:.3f} | {mw:.3f} | "
                     f"{mw - mr:+.3f} | {len(right)} |")
    lines.append("")

    lines += ["## settle step by task complexity", "",
              "| hops | loops | mean settle step | mean experts/token | n |",
              "| --- | --- | --- | --- | --- |"]
    buckets = {}
    for name, run, traj in _rows():
        hops = run["spec"]["hops"][0]
        key = (hops, run["config"]["loops"])
        if f"settle_h{hops}" in traj:
            buckets.setdefault(key, []).append(
                (traj[f"settle_h{hops}"], traj.get(f"mean_k_h{hops}", float("nan"))))
    for (hops, loops), vals in sorted(buckets.items()):
        settle, _ = mean_sd([v[0] for v in vals])
        ks, _ = mean_sd([v[1] for v in vals])
        lines.append(f"| {hops} | {loops} | {settle:.3f} | {ks:.2f} | {len(vals)} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true",
                    help="accepted for symmetry; this experiment never trains")
    ap.parse_args()
    print(report())


if __name__ == "__main__":
    main()
