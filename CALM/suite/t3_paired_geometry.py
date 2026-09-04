#!/usr/bin/env python3
"""T3: euclidean vs Lorentz head geometry, as a PAIRED experiment.

T1 and T2 both compared arm *means* and drowned. With a single-run seed sd of
5.04 and effects around 1.6 perplexity, separating two arms by their means
needs ~79 seeds each -- 16 hours of CPU. That is why P1 and P2 came back null:
the design could not answer its own question, at any budget we have.

But the geometry comparison does not need that, because it can be paired
**exactly**:

* the lift to the hyperboloid is a *function*, not a layer, so the two arms
  have identical parameter names, shapes and counts (verified: 6,135,936 each,
  bit-identical tensors at a shared seed);
* the data stream seed is fixed, so both arms see the same batches in the same
  order;
* there is no dropout.

So for a given seed the two arms differ by exactly one thing: whether the
attention score is a dot product or a Minkowski inner product. The per-seed
*difference* therefore cancels the initialisation variance that was the entire
noise floor, and the right statistic is the mean paired difference and its
own standard deviation -- not the difference of two arm means.

**The evidence that motivated this.** Every euclid/lorentz pair collected so
far, across four different learning rates, went the same way:

    lr 5e-4   265.28 vs 268.58   +3.30
    lr 1e-3   231.13 vs 232.85   +1.72
    lr 2e-3   215.00 vs 216.96   +1.96
    lr 1e-3   198.24 vs 199.72   +1.48

Four of four, same sign, magnitudes 1.5-3.3. Reported as arm means those look
like "0.7%, within noise". Read as paired differences they are a consistent
small penalty for the Lorentz score, and this experiment is built to find out
whether that survives more seeds.

**Prediction, registered before the run.** `WHY_HYPERBOLIC.md` argues the
mechanism is dimension efficiency, so any hyperbolic advantage should be
largest at small head_dim and shrink as head_dim grows. The paired data above
points the other way -- a small, consistent *penalty*. If that holds, the
hyperbolic head-space hypothesis is dead in this regime and the interesting
question becomes why the penalty is so stable.

Usage::

    python CALM/suite/t3_paired_geometry.py --seeds 0,1,2,3
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
from t1_head_geometry import build, perplexity, train  # noqa: E402


def assert_paired(vocab, options, head_dim):
    """Refuse to run unless the two arms really are bit-identical at init.

    The whole design rests on this. If a future change to the model adds a
    parameter on one side -- a learnable curvature, a wider up-projection --
    the pairing silently breaks and the paired statistic becomes wrong rather
    than merely noisy. So it is checked, not assumed.
    """
    made = []
    for geometry in ("euclidean", "lorentz"):
        torch.manual_seed(12345)
        made.append(dict(build(vocab, options.dim, options.layers, head_dim,
                               options.kv_latent, geometry, "euclidean",
                               options.seq_len).named_parameters()))
    a, b = made
    if set(a) != set(b) or not all(torch.equal(a[k], b[k]) for k in a):
        raise SystemExit(
            f"PAIRING BROKEN at head_dim {head_dim}: the euclidean and lorentz "
            f"arms are not bit-identical at initialisation, so a paired "
            f"comparison is invalid. Fix the model or use an unpaired design.")


def sign_test_p(diffs: List[float]) -> float:
    """Two-sided exact sign test. Small n, no distributional assumption."""
    from math import comb
    n = sum(1 for d in diffs if d != 0)
    k = sum(1 for d in diffs if d > 0)
    if n == 0:
        return 1.0
    k = max(k, n - k)
    tail = sum(comb(n, i) for i in range(k, n + 1))
    return min(1.0, 2 * tail / 2 ** n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--kv-latent", type=int, default=48)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--vocab", type=int, default=16000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--head-dims", default="16,32")
    parser.add_argument("--seeds", default="0,1,2,3")
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
    head_dims = [int(v) for v in options.head_dims.split(",")]
    seeds = [int(v) for v in options.seeds.split(",")]

    print(f"T3  PAIRED euclidean vs lorentz head geometry")
    print(f"    wikitext2 BPE {vocab}, dim {options.dim}, {options.layers} layers, "
          f"kv_latent {options.kv_latent}, lr {options.lr:.1e}")
    print(f"    {options.steps} steps, head_dims {head_dims}, "
          f"{len(seeds)} seeds -> {len(seeds)} paired differences per head_dim")
    print("    statistic: the per-seed DIFFERENCE, which cancels the "
          "initialisation\n    variance that made T1/T2 unresolvable.\n")

    for head_dim in head_dims:
        assert_paired(vocab, options, head_dim)
    print(f"    pairing verified: arms bit-identical at init for {head_dims}\n")

    rows: List[Dict] = []
    print(f"{'head_dim':>9s} {'seed':>5s} {'euclidean':>10s} {'lorentz':>10s} "
          f"{'lorentz - euclid':>17s}")
    for head_dim in head_dims:
        for seed in seeds:
            scores = {}
            for geometry in ("euclidean", "lorentz"):
                torch.manual_seed(seed)
                model = build(vocab, options.dim, options.layers, head_dim,
                              options.kv_latent, geometry, "euclidean",
                              options.seq_len).to(device)
                stream = stream_from(train_ids, options.batch, options.seq_len, 0)
                model, _ = train(model, stream, options.steps, options.lr, device)
                scores[geometry] = perplexity(model, eval_batches)
            diff = scores["lorentz"] - scores["euclidean"]
            rows.append({"head_dim": head_dim, "seed": seed, **scores,
                         "diff": diff})
            print(f"{head_dim:9d} {seed:5d} {scores['euclidean']:10.2f} "
                  f"{scores['lorentz']:10.2f} {diff:+17.2f}", flush=True)
            if options.out:
                Path(options.out).parent.mkdir(parents=True, exist_ok=True)
                Path(options.out).write_text(json.dumps(
                    {"config": vars(options), "rows": rows}, indent=2))
    report(rows, head_dims)


def report(rows, head_dims):
    print("\n" + "=" * 74)
    print("PAIRED ANALYSIS  (positive = lorentz WORSE)")
    print(f"{'head_dim':>9s} {'n':>3s} {'mean diff':>10s} {'sd of diff':>11s} "
          f"{'mean %':>8s} {'signs':>7s} {'sign p':>8s}")
    for head_dim in head_dims:
        diffs = [r["diff"] for r in rows if r["head_dim"] == head_dim]
        base = [r["euclidean"] for r in rows if r["head_dim"] == head_dim]
        if not diffs:
            continue
        mean = statistics.mean(diffs)
        sd = statistics.stdev(diffs) if len(diffs) > 1 else float("nan")
        pos = sum(1 for d in diffs if d > 0)
        print(f"{head_dim:9d} {len(diffs):3d} {mean:+10.2f} {sd:11.2f} "
              f"{mean / statistics.mean(base) * 100:+7.2f}% "
              f"{pos:3d}/{len(diffs):<3d} {sign_test_p(diffs):8.3f}")

    everything = [r["diff"] for r in rows]
    if len(everything) > 1:
        mean = statistics.mean(everything)
        sd = statistics.stdev(everything)
        stderr = sd / len(everything) ** 0.5
        print(f"\n  pooled: mean {mean:+.2f} +- {stderr:.2f} (sem), "
              f"{sum(1 for d in everything if d > 0)}/{len(everything)} positive, "
              f"sign p = {sign_test_p(everything):.4f}")
        # Report a proper t statistic against the right critical value.
        # An earlier version printed "|mean|/sem > 2 is a resolved effect",
        # which is the normal approximation; at n=6 the correct threshold is
        # t(0.975, 5) = 2.571, and applying the loose rule would have declared
        # head_dim 16 resolved at p = 0.081.
        crit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
                7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201}
        df = len(everything) - 1
        print(f"  t = {abs(mean) / stderr:.2f} on {df} df "
              f"(two-sided 5% critical value {crit.get(df, 1.96):.3f})")

    print("\n  Interpretation guide, fixed in advance:")
    print("    - mean diff > 0 and sign p < 0.05  -> lorentz head geometry is")
    print("      reliably WORSE; the per-head hyperbolic hypothesis fails here.")
    print("    - the head_dim trend is the WHY_HYPERBOLIC prediction: an")
    print("      advantage that shrinks as head_dim grows. A penalty that is")
    print("      FLAT in head_dim points at a fixed cost of the lift instead")
    print("      (e.g. the time coordinate spending capacity on |q|), not at")
    print("      a dimension-efficiency effect at all.")


if __name__ == "__main__":
    main()
