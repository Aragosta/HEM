"""A minimal learnability probe: one arm, one hop set, printed as it goes.

Kept in the folder because it is what diagnosed the two dead pilots (sparse
supervision, then an extra composition step in the sequence layout), and
because anyone changing the task should re-run it before spending an
experiment's compute on a task that cannot be learned.

    python probe_learnable.py --hops 1 --loops 1 --steps 400 --lr 3e-3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import answer_logits, evaluate_hops, _lr_at, _optimizer  # noqa: E402
from model import Config, Recurrent                                    # noqa: E402
from tasks import HopSpec, hop_batch, hop_eval_set                     # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hops", default="1")
    ap.add_argument("--loops", type=int, default=1)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--entities", type=int, default=12)
    ap.add_argument("--queries", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--sorted", action="store_true")
    ap.add_argument("--cfg", default="", help="config overrides, k=v,k=v")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    hops = tuple(int(h) for h in args.hops.split(","))
    spec = HopSpec(n_entities=args.entities, hops=hops, queries=args.queries,
                   sorted_pairs=args.sorted)
    overrides = {}
    for item in filter(None, args.cfg.split(",")):
        key, _, value = item.partition("=")
        if value in ("True", "False"):
            overrides[key] = value == "True"
        elif value.replace(".", "", 1).isdigit():
            overrides[key] = float(value) if "." in value else int(value)
        else:
            overrides[key] = value
    cfg = Config(vocab_size=spec.vocab_size, dim=args.dim, n_heads=4,
                 max_seq_len=spec.seq_len, loops=args.loops, seed=0, **overrides)
    model = Recurrent(cfg)
    opt = _optimizer(model, args.lr)
    g = torch.Generator().manual_seed(1234)
    eval_set = hop_eval_set(spec, 128, 99)
    positions = spec.answer_positions()
    print(f"{spec.describe()} | R={args.loops} lr={args.lr} "
          f"chance={1 / spec.n_entities:.3f} | overrides={overrides}", flush=True)

    for step in range(args.steps):
        for group in opt.param_groups:
            group["lr"] = _lr_at(step, args.steps, args.lr)
        tokens, target, _ = hop_batch(spec, args.batch, g)
        logits, _ = model(tokens)
        pred = answer_logits(logits, positions)
        loss = F.cross_entropy(pred.reshape(-1, pred.shape[-1]), target.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.rebalance()
        if (step + 1) % max(1, args.steps // 10) == 0:
            ev = evaluate_hops(model, eval_set, spec)
            print(f"  step {step + 1:5d} loss {loss.item():.3f} "
                  f"acc {ev['acc']:.3f} "
                  + " ".join(f"h{h}={ev[f'acc_h{h}']:.3f}" for h in hops),
                  flush=True)


if __name__ == "__main__":
    main()
