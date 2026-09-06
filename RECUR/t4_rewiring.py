#!/usr/bin/env python3
"""T4: let the expert graph choose its own sparsity, by a purely local rule.

Top-k routing imposes the number of edges. The self-organised-criticality
literature says a network can *find* its operating point without being told it:
add or delete a connection using only that node's own activity, and criticality
is an attractive fixed point of the rewiring rule (arXiv:2009.11781,
arXiv:1203.4942; the same idea imported into deep nets in arXiv:2608.28431).
`CALM/CRITICALITY.md` proposed this for MoE routing as C4 and never built it.

The rule implemented in `LatentMoE.homeostasis` is the minimal version: each
expert has its own threshold, takes any token whose router score clears it, and
raises or lowers that threshold according to whether it has been firing more or
less than its share. Nothing in the rule knows the layer's target sparsity, and
**the number of experts per token is an output, not a setting**.

Two arms, one field apart, at two task complexities:

| arm | `route_mode` | experts per token |
| --- | --- | --- |
| `topk` | `"topk"` | fixed at 2, chosen by us |
| `local` | `"threshold"` | emergent, chosen by the rule |

Registered predictions:

- **T4a.** `local` lands within noise of `topk` on accuracy. The claim in the
  literature is robustness and self-tuning, not raw performance, so a win would
  be a surprise and a large loss would mean the rule is mis-specified.
- **T4b, the interesting one.** The emergent experts-per-token **declines
  across loop positions** -- wide entry, narrow refinement -- reproducing E3's
  phase distinction from a rule that was never told loops exist. If instead it
  is flat, the phase story is an artefact of how we conditioned the router
  rather than a property of the computation.
- **T4c.** The emergent k rises with hop count, agreeing with T1's k* (and with
  arXiv:2410.13964) from a mechanism rather than a sweep. Two independent routes
  to the same number is the strongest evidence this scale can produce.

Usage::

    python t4_rewiring.py --seeds 0,1 --workers 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import load_results, mean_sd, run_jobs                # noqa: E402
from spec import BATCH, HOPS_CFG, LR, hop_spec                    # noqa: E402

HOPS = (2, 3)
STEPS = 2000
ARMS = {"topk": {}, "local": {"route_mode": "threshold"}}


def jobs(seeds, steps: int = STEPS):
    out = []
    for h in HOPS:
        for arm, delta in ARMS.items():
            for seed in seeds:
                out.append({"name": f"t4_h{h}_{arm}_s{seed}", "task": "hops",
                            "spec": hop_spec(h),
                            "cfg": {**HOPS_CFG, "loops": 2, "n_active": 2, **delta},
                            "steps": steps, "batch": BATCH, "lr": LR,
                            "seed": seed})
    return out


def report() -> str:
    lines = ["# T4 -- local rewiring instead of top-k", "",
             "| hops | arm | accuracy | sd | emergent experts/token | load sd |",
             "| --- | --- | --- | --- | --- | --- |"]
    for h in HOPS:
        for arm in ARMS:
            runs = load_results(f"t4_h{h}_{arm}_s")
            if not runs:
                continue
            m, sd = mean_sd([x["final"]["acc"] for x in runs.values()])
            ks, loads = [], []
            for r in runs.values():
                state = r.get("routing_state", {})
                if state.get("mean_k"):
                    ks.append(sum(state["mean_k"]) / len(state["mean_k"]))
                if state.get("load"):
                    last = state["load"][-1]
                    mu = sum(last) / len(last)
                    loads.append((sum((x - mu) ** 2 for x in last) / len(last)) ** 0.5)
            k_txt = f"{sum(ks) / len(ks):.2f}" if ks else "-"
            l_txt = f"{sum(loads) / len(loads):.4f}" if loads else "-"
            lines.append(f"| {h} | {arm} | {m:.3f} | {sd:.3f} | {k_txt} | {l_txt} |")
    lines.append("")

    lines += ["## experts per token by loop position (T4b)", "",
              "From the co-activation record of the final model: the mean "
              "number of experts each token activates at loop 1 and loop 2. A "
              "declining profile is the wide-entry / narrow-refinement pattern "
              "that E3 found by conditioning the router, arrived at here by a "
              "local rule that knows nothing about loops.", "",
              "| hops | arm | seed | loop 1 | loop 2 |",
              "| --- | --- | --- | --- | --- |"]
    for h in HOPS:
        for arm in ARMS:
            for name, r in sorted(load_results(f"t4_h{h}_{arm}_s").items()):
                graph = r.get("expert_graph", {})
                if not graph:
                    continue
                cells = [f"{graph[key]['mean_k']:.2f}" if key in graph else "-"
                         for key in ("loop1", "loop2")]
                lines.append(f"| {h} | {arm} | {name[-1]} | " + " | ".join(cells) + " |")
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
