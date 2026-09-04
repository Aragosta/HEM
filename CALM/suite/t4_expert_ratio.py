#!/usr/bin/env python3
"""T4: does MoE help once the total/active ratio is actually MoE-like?

T2 compared a dense FFN against a 4-expert MoE and found nothing. That was not
a test of the MoE hypothesis. MoE's mechanism is **more parameters at the same
FLOPs**, and T2's ratio was 1.22x:

    ours (4 experts, top-2 + 1 shared)   6,431,616 / 5,251,968 = 1.22x
    Mixtral 8x7B                         46.7B / 12.9B         = 3.6x
    DeepSeek-V3                          671B / 37B            = 18.1x

Asking whether 22% more parameters in routed form helps after 0.3 epochs is
not the same question as whether MoE works, and the null answer to the first
says nothing about the second.

Raising `n_experts` at fixed `top_k` is nearly free in compute -- the router is
a dim x n_experts matmul and the number of experts *evaluated per token* does
not change -- so the ratio can be swept on CPU at constant active FLOPs:

    n_experts  4 -> 1.22x   357 ms/step
    n_experts 16 -> 2.57x   404 ms/step
    n_experts 32 -> 4.36x   553 ms/step

(the wall-clock growth is Python loop overhead in MoE.forward, not arithmetic;
active parameters stay within 0.4% across the sweep.)

**Registered prediction.** If MoE's benefit comes from the parameter/compute
ratio, perplexity should fall monotonically as the ratio rises, with dense
(1.0x) worst. If all four arms land together, then at this scale and token
budget the extra capacity cannot be used at all, and the honest conclusion is
that MoE is untestable here rather than unhelpful.

**Why a trend, not a pairwise test.** Dense and MoE cannot be paired -- they
have different parameter counts, so no shared initialisation exists. The
unpaired seed sd measured in T2 was 5.04, and separating two arms by their
means would need ~79 seeds each. A *monotone trend across four ratios* is
detectable where a single pairwise contrast is not: it uses all eight runs
jointly and asks for a slope rather than a gap.

**Two known confounds this does NOT fix**, both flagged so the result is not
over-read:

1. the embedding table is 58% of the dense model at vocab 16000 x dim 192, so
   every FFN change is diluted into roughly a third of the parameters;
2. 1000 steps is 0.43 epochs, and expert specialisation is slow. Experts may
   simply not have had time to differentiate.

Either could produce a flat result independently of the ratio hypothesis.

Usage::

    python CALM/suite/t4_expert_ratio.py --steps 1000 --seeds 0,1
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List

import torch

ROOT = Path(__file__).resolve().parents[2]
for _extra in (ROOT, ROOT / "CALM", ROOT / "CALM" / "suite"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from bpe import encode_split, train_or_load  # noqa: E402
from corpus import batches_from, stream_from  # noqa: E402
from hybrid import HybridDecoder, MoE  # noqa: E402
from t1_head_geometry import perplexity, train  # noqa: E402
from t2_criticality import active_params, measure  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--kv-latent", type=int, default=48)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--vocab", type=int, default=16000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--n-shared", type=int, default=1)
    parser.add_argument("--experts", default="4,16,32",
                        help="n_experts values; dense is always included")
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--out", default="")
    options = parser.parse_args()

    device = torch.device(options.device)
    tokenizer = train_or_load("wikitext2", options.vocab)
    train_ids = encode_split(tokenizer, "wikitext2", "train")
    valid_ids = encode_split(tokenizer, "wikitext2", "valid")
    vocab = tokenizer.get_vocab_size()
    eval_batches = [b.to(device) for b in batches_from(
        valid_ids, options.batch, options.seq_len, options.eval_batches, seed=1)]
    experts = [int(v) for v in options.experts.split(",")]
    seeds = [int(v) for v in options.seeds.split(",")]
    arms = [("dense", 0)] + [("moe", n) for n in experts]
    seen = options.steps * options.batch * options.seq_len

    print("T4  does MoE help once total/active is actually MoE-like?")
    print(f"    wikitext2 BPE {vocab}, dim {options.dim}, {options.layers} layers, "
          f"lr {options.lr:.1e}")
    print(f"    {options.steps} steps = {seen:,} tokens = "
          f"{seen / train_ids.numel():.2f} epochs")
    print(f"    top_k {options.top_k} + {options.n_shared} shared held fixed, so "
          f"ACTIVE flops are constant\n    across the sweep; only total capacity "
          f"grows.")
    print("    prediction: perplexity falls monotonically as total/active rises.")
    print("    if all arms land together, MoE is untestable at this scale, not "
          "unhelpful.\n")

    def make(kind, n_experts, seed):
        torch.manual_seed(seed)
        return HybridDecoder(
            vocab, dim=options.dim, layers=options.layers,
            heads=max(options.dim // options.head_dim, 1),
            head_dim=options.head_dim, kv_latent=options.kv_latent,
            ffn=kind, n_experts=max(n_experts, 1), top_k=options.top_k,
            n_shared=options.n_shared, max_seq_len=options.seq_len).to(device)

    rows: List[Dict] = []
    print(f"{'arm':>10s} {'ratio':>7s} {'seed':>5s} {'perplexity':>11s} "
          f"{'total':>11s} {'active':>11s} {'load_bal':>9s} {'dead':>5s}")
    for kind, n_experts in arms:
        for seed in seeds:
            model = make(kind, n_experts, seed)
            total = sum(p.numel() for p in model.parameters())
            active = active_params(model)
            stream = stream_from(train_ids, options.batch, options.seq_len, 0)
            model, per_step = train(model, stream, options.steps, options.lr,
                                    device)
            ppl = perplexity(model, eval_batches)
            stats = measure(model, eval_batches[:4])
            label = "dense" if kind == "dense" else f"moe{n_experts}"
            row = {"arm": label, "n_experts": n_experts, "seed": seed,
                   "perplexity": ppl, "total": total, "active": active,
                   "ratio": total / active, "ms_per_step": per_step * 1000,
                   **stats}
            rows.append(row)
            print(f"{label:>10s} {total/active:6.2f}x {seed:5d} {ppl:11.2f} "
                  f"{total:11,} {active:11,} "
                  f"{stats.get('load_balance', float('nan')):9.4f} "
                  f"{stats.get('dead_experts', float('nan')):5.1f}", flush=True)
            if options.out:
                Path(options.out).parent.mkdir(parents=True, exist_ok=True)
                Path(options.out).write_text(json.dumps(
                    {"config": vars(options), "rows": rows}, indent=2))
    report(rows)


def report(rows):
    import math
    labels = []
    for r in rows:
        if r["arm"] not in labels:
            labels.append(r["arm"])

    print("\n" + "=" * 72)
    print(f"{'arm':>10s} {'ratio':>7s} {'mean ppl':>10s} {'sd':>7s} "
          f"{'load_bal':>9s} {'dead':>5s}")
    points = []
    for label in labels:
        group = [r for r in rows if r["arm"] == label]
        ppl = [r["perplexity"] for r in group]
        ratio = group[0]["ratio"]
        mean = statistics.mean(ppl)
        sd = statistics.stdev(ppl) if len(ppl) > 1 else 0.0
        lb = [r["load_balance"] for r in group if "load_balance" in r]
        dead = [r["dead_experts"] for r in group if "dead_experts" in r]
        print(f"{label:>10s} {ratio:6.2f}x {mean:10.2f} {sd:7.2f} "
              f"{statistics.mean(lb) if lb else float('nan'):9.4f} "
              f"{statistics.mean(dead) if dead else float('nan'):5.1f}")
        for value in ppl:
            points.append((math.log(ratio), value))

    if len(points) > 2:
        # Least-squares slope of perplexity on log(total/active), using every
        # individual run rather than the arm means, so the seed spread is
        # inside the residual and the standard error is honest.
        n = len(points)
        mx = sum(x for x, _ in points) / n
        my = sum(y for _, y in points) / n
        sxx = sum((x - mx) ** 2 for x, _ in points)
        sxy = sum((x - mx) * (y - my) for x, y in points)
        slope = sxy / sxx if sxx else float("nan")
        intercept = my - slope * mx
        resid = [y - (intercept + slope * x) for x, y in points]
        se_slope = (sum(r * r for r in resid) / (n - 2) / sxx) ** 0.5 if sxx else float("nan")
        t = slope / se_slope if se_slope else float("nan")
        crit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
                7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}.get(n - 2, 1.96)
        print(f"\n  slope of perplexity on log(total/active):")
        print(f"    {slope:+.2f} perplexity per e-fold, se {se_slope:.2f}, "
              f"t = {t:+.2f} on {n-2} df (5% crit {crit:.3f})")
        if abs(t) < crit:
            print("    -> NOT resolved. No detectable benefit from raising the")
            print("       parameter/compute ratio at this scale and token budget.")
            print("       Read with the two confounds in the module docstring:")
            print("       the embedding is ~58% of the model, and 0.43 epochs")
            print("       may be too few for experts to specialise.")
        elif slope < 0:
            print("    -> resolved and NEGATIVE: perplexity falls as the ratio")
            print("       rises, which is the MoE mechanism working as claimed.")
        else:
            print("    -> resolved and POSITIVE: more total capacity makes it")
            print("       WORSE, which would point at an optimisation problem")
            print("       (under-trained experts) rather than a capacity one.")


if __name__ == "__main__":
    main()
