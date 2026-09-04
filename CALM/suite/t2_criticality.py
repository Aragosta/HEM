#!/usr/bin/env python3
"""T2: dense MHA vs MoE MHA, and does making the attention temperature
learnable buy anything?

`CRITICALITY.md` argues that a transformer layer is already a dense graph
(attention) annealed at an inverse temperature `beta`, plus -- in the MoE case
-- a second, *discrete* graph reduction over the expert graph. The theory
(arXiv:2510.05554) says the softmax graph has three phases and only the middle
one is useful, and that the standard `beta = 1/sqrt(d_head)` is a variance
argument rather than an optimality argument.

This experiment tests the two cheapest consequences and instruments the order
parameters throughout, on a 2x2:

    ffn        in {dense, moe}      -- is routing worth it at matched FLOPs?
    beta_mode  in {fixed, learned}  -- is 1/sqrt(d) the right temperature?

**Matched FLOPs.** The MoE arm activates `top_k + n_shared` experts of width
`inter / (top_k + n_shared)`, so both arms do the same work per token and
differ only in whether that work is *routed*. Without this the comparison
silently becomes a capacity comparison. The MoE arm still has more *total*
parameters -- that is what MoE is for -- and both counts are reported.

**Zero-cost beta.** `learned` adds one scalar per head (8 parameters in the
smoke configuration, ~24 here) initialised at exactly `log(1/sqrt(d))`, so at
step 0 the two beta arms are bit-identical. Any difference is the temperature
moving, nothing else.

**Predictions, registered before the run:**

P1. At matched active FLOPs the MoE arm wins by a small margin or ties. If it
    *loses*, the routing is not paying for its overhead at this scale and the
    dense baseline is the honest default.
P2. `learned` beats `fixed` in both FFN arms, and the learned beta moves
    **up** (sharper attention, beta_ratio > 1). Reason: at initialisation the
    measured order parameters are `entropy_norm = 1.00` and
    `participation_frac = 1.00` -- the model starts fully in the disordered
    phase, so the useful direction is toward more concentration.
P3. Final perplexity tracks the order parameters: arms that end with lower
    `participation_frac` (further from the disordered phase) score better. If
    perplexity moves and the order parameters do not, the phase-transition
    framing is not describing what training actually does here.

P2 is the one that matters beyond this experiment. If a free scalar per head is
worth real perplexity, then T0's hyperbolic collapse (269 vs 86) is a candidate
temperature bug rather than a verdict on geometry -- the Lorentz arm inherits
`1/sqrt(d+1)` from an algebra whose inner products have a different scale.

Usage::

    python CALM/suite/t2_criticality.py --time-only
    python CALM/suite/t2_criticality.py --steps 2000 --lr-sweep 1e-3,2e-3,4e-3
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
from t1_head_geometry import perplexity, train  # noqa: E402

ARMS = [(ffn, beta) for ffn in ("dense", "moe")
        for beta in ("fixed", "learned")]


def build(vocab, options, ffn, beta_mode):
    heads = max(options.dim // options.head_dim, 1)
    return HybridDecoder(vocab, dim=options.dim, layers=options.layers,
                         heads=heads, head_dim=options.head_dim,
                         kv_latent=options.kv_latent, ffn=ffn,
                         beta_mode=beta_mode, top_k=options.top_k,
                         n_experts=options.n_experts,
                         n_shared=options.n_shared,
                         max_seq_len=options.seq_len)


def active_params(model) -> int:
    """Parameters actually touched by one token.

    Everything except the experts that top-k routing did not select. Reported
    beside the total so that a win cannot be quietly attributed to extra
    compute the other arm never had.
    """
    from hybrid import MoE
    total = sum(p.numel() for p in model.parameters())
    for module in model.modules():
        if isinstance(module, MoE):
            skipped = module.n_experts - module.top_k
            per_expert = sum(p.numel() for p in module.experts[0].parameters())
            total -= skipped * per_expert
    return total


@torch.no_grad()
def measure(model, batches) -> Dict[str, float]:
    """Order parameters, averaged over a few batches."""
    model.set_stats(True)
    runs: Dict[str, List[float]] = {}
    for tokens in batches:
        model.logits(tokens[:, :-1])
        for key, value in model.order_parameters().items():
            if key != "per_layer":
                runs.setdefault(key, []).append(value)
    model.set_stats(False)
    return {k: statistics.mean(v) for k, v in runs.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--kv-latent", type=int, default=48)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--vocab", type=int, default=16000)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--n-experts", type=int, default=4)
    parser.add_argument("--n-shared", type=int, default=1)
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-batches", type=int, default=24)
    parser.add_argument("--stat-batches", type=int, default=4)
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
    seeds = [int(v) for v in options.seeds.split(",")]
    seen = options.steps * options.batch * options.seq_len

    print(f"T2  wikitext2 BPE {vocab}, dim {options.dim}, {options.layers} layers, "
          f"head_dim {options.head_dim}, kv_latent {options.kv_latent}")
    print(f"    {options.steps} steps = {seen:,} tokens = "
          f"{seen / train_ids.numel():.2f} epochs, seq {options.seq_len}")
    print(f"    MoE: {options.n_experts} experts, top-{options.top_k}, "
          f"{options.n_shared} shared -> FLOP-matched to the dense FFN")
    print("    predictions: P1 MoE >= dense at matched FLOPs; P2 learned beta "
          "wins and\n    moves up; P3 perplexity tracks the order parameters.\n")

    def make(ffn, beta_mode, seed):
        torch.manual_seed(seed)
        return build(vocab, options, ffn, beta_mode).to(device)

    if options.time_only:
        for ffn, beta_mode in ARMS:
            model = make(ffn, beta_mode, 0)
            stream = stream_from(train_ids, options.batch, options.seq_len, 0)
            _, per_step = train(model, stream, 8, options.lr, device)
            print(f"  {ffn:6s} {beta_mode:8s} {per_step*1000:7.1f} ms/step -> "
                  f"{options.steps} steps = {per_step*options.steps/60:5.1f} min "
                  f"| total {sum(p.numel() for p in model.parameters()):,} "
                  f"active {active_params(model):,}")
        return

    if options.lr_sweep:
        # One shared grid: all four arms use plain AdamW on Euclidean
        # parameters, so a shared learning rate is not the confound it was in
        # T0. The grid is scored on the mean over arms so no single arm's
        # optimum is imposed on the others without being seen.
        print("LR sweep (shared AdamW across all arms, so a shared grid is fair):")
        best, rates = None, [float(v) for v in options.lr_sweep.split(",")]
        for rate in rates:
            scores = []
            for ffn, beta_mode in ARMS:
                model = make(ffn, beta_mode, 0)
                stream = stream_from(train_ids, options.batch, options.seq_len, 0)
                model, _ = train(model, stream, options.sweep_steps, rate, device)
                scores.append(perplexity(model, eval_batches[:8]))
            print("  lr {:8.1e}  ".format(rate) + "  ".join(
                f"{a[0][:3]}/{a[1][:3]} {s:7.2f}" for a, s in zip(ARMS, scores)),
                flush=True)
            mean = statistics.mean(scores)
            if best is None or mean < best[1]:
                best = (rate, mean)
        if best[0] in (min(rates), max(rates)) and len(rates) > 1:
            raise SystemExit(f"\nSWEEP FAILED: best lr {best[0]:.1e} is at the "
                             f"edge of the grid; widen it.")
        options.lr = best[0]
        print(f"  -> using lr {options.lr:.1e}\n")

    rows: List[Dict] = []
    print(f"{'ffn':>6s} {'beta':>8s} {'seed':>5s} {'perplexity':>11s} "
          f"{'part_frac':>10s} {'entropy':>8s} {'beta_x':>7s} {'load_bal':>9s}")
    for ffn, beta_mode in ARMS:
        for seed in seeds:
            model = make(ffn, beta_mode, seed)
            stream = stream_from(train_ids, options.batch, options.seq_len, 0)
            model, per_step = train(model, stream, options.steps, options.lr,
                                    device)
            ppl = perplexity(model, eval_batches)
            stats = measure(model, eval_batches[:options.stat_batches])
            row = {"ffn": ffn, "beta_mode": beta_mode, "seed": seed,
                   "perplexity": ppl, "ms_per_step": per_step * 1000,
                   "params": sum(p.numel() for p in model.parameters()),
                   "active_params": active_params(model), **stats}
            rows.append(row)
            print(f"{ffn:>6s} {beta_mode:>8s} {seed:5d} {ppl:11.2f} "
                  f"{stats.get('participation_frac', float('nan')):10.4f} "
                  f"{stats.get('entropy_norm', float('nan')):8.4f} "
                  f"{stats.get('beta_ratio', float('nan')):7.3f} "
                  f"{stats.get('load_balance', float('nan')):9.4f}", flush=True)
            if options.out:
                Path(options.out).parent.mkdir(parents=True, exist_ok=True)
                Path(options.out).write_text(json.dumps(
                    {"config": vars(options), "rows": rows}, indent=2))
    report(rows)


def report(rows):
    def group(ffn, beta_mode):
        return [r for r in rows
                if r["ffn"] == ffn and r["beta_mode"] == beta_mode]

    def mean(rs, key):
        values = [r[key] for r in rs if key in r]
        return statistics.mean(values) if values else float("nan")

    print("\n" + "=" * 78)
    print(f"{'arm':>15s} {'perplexity':>11s} {'seed sd':>9s} "
          f"{'total params':>13s} {'active':>11s}")
    for ffn, beta_mode in ARMS:
        rs = group(ffn, beta_mode)
        if not rs:
            continue
        ppl = [r["perplexity"] for r in rs]
        sd = statistics.stdev(ppl) if len(ppl) > 1 else 0.0
        print(f"{ffn + '/' + beta_mode:>15s} {statistics.mean(ppl):11.2f} "
              f"{sd:9.2f} {mean(rs, 'params'):13,.0f} "
              f"{mean(rs, 'active_params'):11,.0f}")

    print("\nP1  routing at matched FLOPs")
    for beta_mode in ("fixed", "learned"):
        d, m = group("dense", beta_mode), group("moe", beta_mode)
        if d and m:
            dp, mp = mean(d, "perplexity"), mean(m, "perplexity")
            verdict = "MoE better" if mp < dp else "dense better"
            print(f"    beta={beta_mode:8s} dense {dp:8.2f}  moe {mp:8.2f}  "
                  f"-> {verdict} by {abs(dp - mp) / dp * 100:5.2f}%")

    print("\nP2  learnable temperature")
    for ffn in ("dense", "moe"):
        f, l = group(ffn, "fixed"), group(ffn, "learned")
        if f and l:
            fp, lp = mean(f, "perplexity"), mean(l, "perplexity")
            ratio = mean(l, "beta_ratio")
            direction = ("sharper" if ratio > 1.01 else
                         "flatter" if ratio < 0.99 else "unmoved")
            verdict = "learned better" if lp < fp else "fixed better"
            print(f"    {ffn:6s} fixed {fp:8.2f}  learned {lp:8.2f}  "
                  f"-> {verdict} by {abs(fp - lp) / fp * 100:5.2f}%; "
                  f"beta x{ratio:.3f} ({direction})")

    print("\nP3  do the order parameters move with perplexity?")
    print(f"    {'arm':>15s} {'part_frac':>10s} {'entropy':>9s} "
          f"{'router_ent':>11s} {'load_bal':>9s} {'dead':>5s}")
    for ffn, beta_mode in ARMS:
        rs = group(ffn, beta_mode)
        if not rs:
            continue
        print(f"    {ffn + '/' + beta_mode:>15s} "
              f"{mean(rs, 'participation_frac'):10.4f} "
              f"{mean(rs, 'entropy_norm'):9.4f} "
              f"{mean(rs, 'router_entropy_norm'):11.4f} "
              f"{mean(rs, 'load_balance'):9.4f} "
              f"{mean(rs, 'dead_experts'):5.1f}")
    print("\n    At initialisation both read 1.0000 (fully disordered: every "
          "token\n    attends uniformly). Distance from 1.0 is how far training "
          "moved the\n    model out of the disordered phase.")


if __name__ == "__main__":
    main()
