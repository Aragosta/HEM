"""Scale-setting pilot: is the composition task learnable, and does depth move it?

Run before any experiment. It answers the two questions that decide whether the
rest of the suite means anything at a size four CPU cores can afford: (a) does
the model learn the task at all, and (b) is there daylight between R=1 and R=4?
If (b) is no, every later depth result would be measuring a task that cannot
show depth, and the honest move is to change the task, not to report a null.

The first pilot answered (a) with *no* -- one supervised token per 36 left the
model at ln(V) after 800 steps at every depth. That is recorded in `RESULTS.md`
because it is the reason the task now asks eight questions per context rather
than one.

Usage::

    python pilot.py --loops 4 --lr 2e-3 --steps 800
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import save, train_hops                       # noqa: E402
from model import Config                                   # noqa: E402
from tasks import HopSpec                                  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loops", type=int, default=4)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--entities", type=int, default=12)
    ap.add_argument("--queries", type=int, default=8)
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    spec = HopSpec(n_entities=args.entities, hops=(1, 2, 3, 4),
                   queries=args.queries)
    cfg = Config(vocab_size=spec.vocab_size, dim=args.dim, n_heads=4,
                 max_seq_len=spec.seq_len, loops=args.loops, seed=args.seed)
    print(spec.describe(), flush=True)
    result, _ = train_hops(cfg, spec, steps=args.steps, batch_size=args.batch,
                           lr=args.lr, eval_every=max(1, args.steps // 8))
    tag = args.tag or f"R{args.loops}_lr{args.lr:g}_s{args.seed}"
    save(f"pilot_{tag}", result)
    print(json.dumps({"loops": args.loops, "lr": args.lr,
                      "seconds": round(result["seconds"]),
                      "final": {k: round(v, 3) for k, v in result["final"].items()
                                if "_r" not in k},
                      "history": [{k: round(v, 3) for k, v in h.items()
                                   if k in ("step", "train_loss", "acc")}
                                  for h in result["history"]]}, indent=1),
          flush=True)


if __name__ == "__main__":
    main()
