#!/usr/bin/env python3
"""T5: the expert co-activation graph -- is it modular, and does structure predict accuracy?

Analysis only; reads the graph each training run records at the end.

The graph: nodes are experts, and an edge between two experts is weighted by how
often they are selected for the same token. Three read-outs per loop, from
`harness.expert_graph`:

* **spectral gap** of the normalised Laplacian -- how well connected the graph is;
* **modularity** of its spectral two-way split -- whether experts form communities;
* **effective experts** `exp(H)` -- how many are really being used, as opposed to
  how many exist.

Why this is worth a table rather than a footnote: E3 produced an oddity that
raw divergence cannot explain. The `embed` conditioner made routing diverge
across loops **four times** as much as `bias` and scored *worse* (0.592 vs
0.607). Two structures produce high divergence -- a graph that splits into
coherent communities, and a graph that shatters -- and only the first should
help.

Registered predictions:

- **T5a.** Accuracy correlates with modularity (or the spectral gap) across
  runs, and not with raw divergence. That would say the loop wants the expert
  graph *partitioned*, not merely *different*.
- **T5b.** Effective experts is well below the nominal count, per the pruning
  literature's finding of heavy redundancy (arXiv:2407.00945 keeps ~4 experts
  with improved performance). If `exp(H)` sits far under `n_routed`, most of the
  expert pool is decoration at this scale and T1's k* is the number that matters.
- **T5c.** Modularity rises from loop 1 to loop 2 under `phase`/`bias`-style
  conditioning and falls under `embed`, which is the structural version of E3's
  result.

Usage::

    python t5_expert_graph.py
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import load_results, mean_sd                          # noqa: E402

FAMILIES = ("t1_", "t3_", "t4_", "e3_")


def _rows():
    for family in FAMILIES:
        for name, run in sorted(load_results(family).items()):
            graph = run.get("expert_graph")
            if graph:
                yield name, run, graph


def _corr(xs, ys):
    if len(xs) < 3:
        return float("nan")
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    return num / den if den else float("nan")


def report() -> str:
    lines = ["# T5 -- the expert co-activation graph", ""]

    lines += ["## structure per loop (first 14 runs)", "",
              "| run | loop | spectral gap | modularity | effective experts | experts/token |",
              "| --- | --- | --- | --- | --- | --- |"]
    for i, (name, run, graph) in enumerate(_rows()):
        if i >= 14:
            break
        for key in sorted(graph):
            g = graph[key]
            lines.append(f"| {name} | {key} | {g['spectral_gap']:.3f} | "
                         f"{g['modularity']:.3f} | {g['effective_experts']:.2f} | "
                         f"{g['mean_k']:.2f} |")
    lines.append("")

    acc, gap, mod, eff = [], [], [], []
    for name, run, graph in _rows():
        final = run.get("final", {})
        if "acc" not in final or not graph:
            continue
        last = graph[sorted(graph)[-1]]
        acc.append(final["acc"])
        gap.append(last["spectral_gap"])
        mod.append(last["modularity"])
        eff.append(last["effective_experts"])
    if acc:
        lines += ["## does structure predict accuracy? (T5a)", "",
                  f"- runs compared: {len(acc)}",
                  f"- corr(accuracy, spectral gap) = {_corr(gap, acc):+.3f}",
                  f"- corr(accuracy, modularity) = {_corr(mod, acc):+.3f}",
                  f"- corr(accuracy, effective experts) = {_corr(eff, acc):+.3f}",
                  "",
                  f"Effective experts: mean {mean_sd(eff)[0]:.2f} of a pool of 16 "
                  f"(T5b).", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true",
                    help="accepted for symmetry; this experiment never trains")
    ap.parse_args()
    print(report())


if __name__ == "__main__":
    main()
