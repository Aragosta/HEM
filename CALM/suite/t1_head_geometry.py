#!/usr/bin/env python3
"""T1: does hyperbolic geometry help where the dimension count says it should?

`WHY_HYPERBOLIC.md` argues that hyperbolic geometry buys **dimension
efficiency** -- on the FlyWire connectome a 2D hyperbolic embedding beats the
fly's own 3D anatomy, but Euclidean overtakes it by d=8-16 and wins outright by
d=32-128. A residual stream is 100-1000x past that crossover, which is why
applying it there (as HELM does) loses. A **per-head attention space** at
d_head = 16-128 is not.

**The falsifiable prediction, stated before the run:** if the mechanism is
dimension efficiency, the hyperbolic head space should help most at small
d_head and the advantage should shrink or vanish as d_head grows. A uniform
effect across d_head means the mechanism is something else and the argument in
`WHY_HYPERBOLIC.md` is wrong. A uniform *absence* means it does not transfer
from connectomes to attention at all.

**Why this comparison is cleaner than T0.** T0's arms differed in optimizer
(`RiemannianAdam` vs `AdamW`), which forced a 16x learning-rate gap and
dominated the result. Here the geometry adds **zero parameters** and **no
manifold parameters** -- the lift to the hyperboloid is a function, not a layer,
and the Lorentz inner product is a dot product on (d_head + 1)-vectors. Both
arms are identical Euclidean-parameter models trained by identical AdamW. The
only difference is one extra coordinate inside the attention score.

Total width `heads * head_dim` is held fixed, so the arms have matched capacity
and matched FLOPs at every point of the sweep.

Usage::

    python CALM/suite/t1_head_geometry.py --time-only
    python CALM/suite/t1_head_geometry.py --steps 3000 --head-dims 16,32,64
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

ROOT = Path(__file__).resolve().parents[2]
for _extra in (ROOT, ROOT / "CALM", ROOT / "CALM" / "suite"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from bpe import encode_split, train_or_load  # noqa: E402
from corpus import batches_from, stream_from  # noqa: E402
from hybrid import HybridDecoder  # noqa: E402


def build(vocab, dim, layers, head_dim, kv_latent, head_geometry,
          latent_geometry, seq_len):
    heads = max(dim // head_dim, 1)
    return HybridDecoder(vocab, dim=dim, layers=layers, heads=heads,
                         head_dim=head_dim, kv_latent=kv_latent,
                         head_geometry=head_geometry,
                         latent_geometry=latent_geometry, max_seq_len=seq_len)


def train(model, stream, steps, lr, device, clip=1.0):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    warmup = max(int(steps * 0.03), 1)

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(steps - warmup, 1)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    model.train()
    started = time.time()
    for _ in range(steps):
        loss = model.loss(next(stream).to(device))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, clip)
        optimizer.step()
        schedule.step()
    return model.eval(), (time.time() - started) / max(steps, 1)


@torch.no_grad()
def perplexity(model, batches):
    nats = count = 0.0
    for tokens in batches:
        logits = model.logits(tokens[:, :-1]).float()
        target = tokens[:, 1:].reshape(-1)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), target)
        nats += loss.item() * target.numel()
        count += target.numel()
    return math.exp(nats / count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--kv-latent", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=192)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--vocab", type=int, default=16000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--head-dims", default="16,32,64")
    parser.add_argument("--latent-geometry", default="euclidean")
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-batches", type=int, default=24)
    parser.add_argument("--lr-sweep", default="")
    parser.add_argument("--sweep-steps", type=int, default=800)
    parser.add_argument("--time-only", action="store_true")
    parser.add_argument("--out", default="")
    options = parser.parse_args()

    device = torch.device(options.device)
    tokenizer = train_or_load("wikitext2", options.vocab)
    train_ids = encode_split(tokenizer, "wikitext2", "train")
    valid_ids = encode_split(tokenizer, "wikitext2", "valid")
    vocab = tokenizer.get_vocab_size()
    eval_batches = [b.to(device) for b in batches_from(
        valid_ids, options.batch, options.seq_len, options.eval_batches, seed=1)]
    head_dims = [int(v) for v in options.head_dims.split(",")]
    seeds = [int(v) for v in options.seeds.split(",")]
    seen = options.steps * options.batch * options.seq_len

    print(f"T1  wikitext2 BPE {vocab}, dim {options.dim}, {options.layers} layers, "
          f"kv_latent {options.kv_latent}")
    print(f"    {options.steps} steps = {seen:,} tokens = "
          f"{seen / train_ids.numel():.2f} epochs, seq {options.seq_len}")
    print(f"    head_dim sweep {head_dims}; heads = dim/head_dim so total width "
          f"is fixed at {options.dim}")
    print("    prediction: if the mechanism is dimension efficiency, the "
          "hyperbolic advantage\n    shrinks as head_dim grows.\n")

    def make(head_dim, geometry, seed):
        torch.manual_seed(seed)
        return build(vocab, options.dim, options.layers, head_dim,
                     options.kv_latent, geometry, options.latent_geometry,
                     options.seq_len).to(device)

    if options.time_only:
        for head_dim in head_dims:
            for geometry in ("euclidean", "lorentz"):
                model = make(head_dim, geometry, 0)
                stream = stream_from(train_ids, options.batch, options.seq_len, 0)
                _, per_step = train(model, stream, 8, options.lr, device)
                print(f"  head_dim {head_dim:3d} {geometry:9s} "
                      f"{per_step*1000:7.1f} ms/step -> {options.steps} steps = "
                      f"{per_step*options.steps/60:5.1f} min")
        return

    if options.lr_sweep:
        print("LR sweep (both geometries share an optimizer, so a shared grid "
              "is legitimate here\nunlike T0):")
        best, rates = None, [float(v) for v in options.lr_sweep.split(",")]
        for rate in rates:
            scores = []
            for geometry in ("euclidean", "lorentz"):
                model = make(head_dims[0], geometry, 0)
                stream = stream_from(train_ids, options.batch, options.seq_len, 0)
                model, _ = train(model, stream, options.sweep_steps, rate, device)
                scores.append(perplexity(model, eval_batches[:8]))
            print(f"  lr {rate:8.1e}  euclid {scores[0]:8.2f}  "
                  f"lorentz {scores[1]:8.2f}", flush=True)
            mean = statistics.mean(scores)
            if best is None or mean < best[1]:
                best = (rate, mean)
        if best[0] in (min(rates), max(rates)) and len(rates) > 1:
            raise SystemExit(f"\nSWEEP FAILED: best lr {best[0]:.1e} is at the "
                             f"edge of the grid; widen it.")
        options.lr = best[0]
        print(f"  -> using lr {options.lr:.1e}\n")

    rows: List[Dict] = []
    print(f"{'head_dim':>9s} {'geometry':>10s} {'seed':>5s} {'perplexity':>11s} "
          f"{'ms/step':>9s}")
    for head_dim in head_dims:
        for geometry in ("euclidean", "lorentz"):
            for seed in seeds:
                model = make(head_dim, geometry, seed)
                stream = stream_from(train_ids, options.batch, options.seq_len, 0)
                model, per_step = train(model, stream, options.steps, options.lr,
                                        device)
                ppl = perplexity(model, eval_batches)
                rows.append({"head_dim": head_dim, "geometry": geometry,
                             "seed": seed, "perplexity": ppl,
                             "ms_per_step": per_step * 1000,
                             "params": sum(p.numel() for p in model.parameters())})
                print(f"{head_dim:9d} {geometry:>10s} {seed:5d} {ppl:11.2f} "
                      f"{per_step*1000:9.1f}", flush=True)
                if options.out:
                    Path(options.out).parent.mkdir(parents=True, exist_ok=True)
                    Path(options.out).write_text(json.dumps(
                        {"config": vars(options), "rows": rows}, indent=2))
    report(rows, head_dims)


def report(rows, head_dims):
    print("\n" + "=" * 66)
    print(f"{'head_dim':>9s} {'euclidean':>11s} {'lorentz':>11s} "
          f"{'lorentz better by':>19s} {'seed sd':>9s}")
    deltas = []
    for head_dim in head_dims:
        def group(geometry):
            return [r["perplexity"] for r in rows
                    if r["head_dim"] == head_dim and r["geometry"] == geometry]
        euclid, lorentz = group("euclidean"), group("lorentz")
        if not euclid or not lorentz:
            continue
        e, l = statistics.mean(euclid), statistics.mean(lorentz)
        noise = max(statistics.stdev(euclid) if len(euclid) > 1 else 0.0,
                    statistics.stdev(lorentz) if len(lorentz) > 1 else 0.0)
        deltas.append((head_dim, (e - l) / e, noise))
        print(f"{head_dim:9d} {e:11.2f} {l:11.2f} {100*(e-l)/e:18.2f}% "
              f"{noise:9.2f}")

    print()
    if len(deltas) < 2:
        print("Need at least two head_dim points to test the prediction.")
        return
    small, large = deltas[0], deltas[-1]
    print(f"advantage at head_dim {small[0]}: {100*small[1]:+.2f}%")
    print(f"advantage at head_dim {large[0]}: {100*large[1]:+.2f}%")
    if small[1] > 0 and small[1] > large[1] + 0.005:
        print("\nPREDICTION HELD: the hyperbolic advantage shrinks as head_dim "
              "grows.\nThat is the dimension-efficiency mechanism, and it says "
              "hyperbolic head spaces\nare worth applying where head_dim is "
              "small.")
    elif max(abs(d[1]) for d in deltas) < 0.01:
        print("\nNO EFFECT at any head_dim. The connectome result does not "
              "transfer to attention\nhead spaces, and the WHY_HYPERBOLIC "
              "argument is wrong about this application.")
    elif small[1] < large[1]:
        print("\nPREDICTION FAILED, and reversed: the advantage GROWS with "
              "head_dim. Whatever is\nhappening, it is not dimension "
              "efficiency.")
    else:
        print("\nEffect present but roughly uniform in head_dim. Not the "
              "dimension-efficiency\nmechanism; something else is producing it.")


if __name__ == "__main__":
    main()
