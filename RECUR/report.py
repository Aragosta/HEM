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

SECTIONS = [
    ("e0_baseline.py", e0_baseline.report),
    ("e1_depth_data.py", e1_depth_data.report),
    ("e2_writable_state.py", e2_writable_state.report),
    ("e4_halting.py", e4_halting.report),
    ("e3_step_routing.py", e3_step_routing.report),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()

    path = HERE / "RESULTS.md"
    text = path.read_text()
    for script, fn in SECTIONS:
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
            print(f"marker for {script} not found; skipped", file=sys.stderr)
    if args.stdout:
        print(text)
    else:
        path.write_text(text)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
