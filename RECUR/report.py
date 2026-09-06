#!/usr/bin/env python3
"""Rebuild the tables in `RESULTS.md` from `results/*.json`.

The narrative parts of `RESULTS.md` are written by hand; the tables are not,
because a hand-copied number is a number that can drift from the run that
produced it. Each experiment section is delimited by a marker comment and
replaced in place, so re-running this after more seeds finish updates the
document without touching the prose.

    python report.py            # rewrite RESULTS.md
    python report.py --stdout   # print what it would write
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import e0_baseline, e1_depth_data, e2_writable_state, e3_step_routing, e4_halting  # noqa: E402
import t1_sparsity_depth, t2_trajectories, t2b_depth4, t3_prune_schedule  # noqa: E402
import t4_rewiring, t5_expert_graph                                       # noqa: E402

# (document, script name in the marker, report function)
SECTIONS = [
    ("RESULTS.md", "e0_baseline.py", e0_baseline.report),
    ("RESULTS.md", "e1_depth_data.py", e1_depth_data.report),
    ("RESULTS.md", "e2_writable_state.py", e2_writable_state.report),
    ("RESULTS.md", "e4_halting.py", e4_halting.report),
    ("RESULTS.md", "e3_step_routing.py", e3_step_routing.report),
    ("EXPERTS.md", "t1_sparsity_depth.py", t1_sparsity_depth.report),
    ("EXPERTS.md", "t3_prune_schedule.py", t3_prune_schedule.report),
    ("EXPERTS.md", "t4_rewiring.py", t4_rewiring.report),
    ("EXPERTS.md", "t2_trajectories.py", t2_trajectories.report),
    ("EXPERTS.md", "t2b_depth4.py", t2b_depth4.report),
    ("EXPERTS.md", "t5_expert_graph.py", t5_expert_graph.report),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()

    texts = {}
    for doc, script, fn in SECTIONS:
        text = texts.get(doc) or (HERE / doc).read_text()
        marker = f"<!-- filled by: python {script} --report-only -->"
        body = fn().strip()
        if not body or "| ---" not in body:
            continue
        if all(line.startswith(("|", "#", "")) and "±" not in line
               for line in body.splitlines()) and "0." not in body:
            continue                       # nothing has finished for this one yet
        # drop the report's own H1, the section already has a heading
        body = "\n".join(body.splitlines()[1:]).strip()
        pattern = re.compile(re.escape(marker) + r"(.*?)(?=\n## |\Z)", re.S)
        replacement = marker + "\n\n" + body + "\n\n"
        if pattern.search(text):
            text = pattern.sub(lambda _: replacement, text, count=1)
        else:
            print(f"marker for {script} not found in {doc}; skipped",
                  file=sys.stderr)
        texts[doc] = text
    for doc, text in texts.items():
        if args.stdout:
            print(text)
        else:
            (HERE / doc).write_text(text)
            print(f"wrote {HERE / doc}")


if __name__ == "__main__":
    main()
