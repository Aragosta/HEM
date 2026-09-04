#!/usr/bin/env python3
"""T2-redux: is the attention temperature inert, or was it just redundant?

T2 made `beta` a learnable per-head scalar and found it bought no perplexity,
settling reproducibly at 0.934x the standard `1/sqrt(d)` across four runs. That
was read as "a free scalar per head is not a degree of freedom, it duplicates
one" -- and `PHYSICS.md` sec.5.1 argues the experiment could not have concluded
anything else, because the compensating knob was left in.

The confound. `hybrid.py` takes `q` and `k` straight from their projections
(RoPE only, no QK-norm), so the effective inverse temperature is

    |q| * |k| * beta

and the projections can rescale to undo whatever `beta` does. `RESULTS.md`
recorded the fingerprint of exactly that: the learned arm ended with a *flatter*
temperature (0.934x) and a *more* concentrated attention (`participation_frac`
0.2174 -> 0.2112 dense, 0.2167 -> 0.1985 MoE). Something moved the other way to
more than cancel it.

The instrument. RMS-normalising `q` and `k` per head, with no learnable gain,
pins `|q| = |k| = sqrt(head_dim)` exactly. The logit becomes
`head_dim * cos(q,k) * beta`, so the magnitude route is closed and the only
remaining routes to a sharper distribution are the *angle* and `beta` itself.
There is deliberately no learnable scale: that would restore the confound.

This is a 2x2, dense FFN only -- routing is a different question and T2 already
found it inside the noise here:

    qk_norm    in {off, on}         -- is the compensating magnitude available?
    beta_mode  in {fixed, learned}  -- is 1/sqrt(d) the right temperature?

**Pairing.** All four arms share every weight tensor bit-identically at a given
seed: `qk_norm` adds no parameters at all, and `learned` adds only `log_beta`,
initialised at exactly `log(1/sqrt(d))` so step 0 is unchanged. The data stream
seed is fixed and there is no dropout, so every contrast here is paired, as in
T3 and T5. `assert_paired` enforces it at launch.

**Predictions, registered before the run:**

P1. **The headline.** With QK-norm on, the learned `beta` moves further from
    1.000 than the 0.934x measured without it. Basis: with the magnitude route
    closed, temperature is the only global scale left, so if the model wants a
    different one it must use `beta` to get it. A null here -- `beta_ratio`
    staying near 0.93-1.00 in both columns -- would mean T2's result survives
    the correction and temperature really is inert at this scale.

P2. **The mechanism, and the sharpest test.** With QK-norm OFF, the learned arm
    should show a *larger* `qk_gain` (= |q|*|k|) than the fixed arm at the same
    seed, because that is what cancelling a 0.93x temperature requires. This is
    a paired, same-initialisation comparison of a quantity nothing in the loss
    mentions. If the sign is inconsistent across seeds, the compensation story
    in `PHYSICS.md` sec.5.1 is wrong and should be withdrawn.
    With QK-norm ON, `qk_gain` must equal `head_dim` exactly in every arm; that
    is a correctness check on the instrument, not a result.

P3. **The attractor.** All four arms still land near `participation_frac` 0.20
    and `entropy_norm` 0.41, as all four T2 arms did from an initialisation at
    1.00. If the QK-norm arms land somewhere else, then the "operating phase is
    an attractor" reading of T2 was really a statement about the q/k gain being
    free, and it weakens.

P4. **Perplexity says nothing.** At 500 steps the seed sd is comparable to the
    entire spread between arms; T2's own methodological finding was that the
    order parameters survive a budget cut and the perplexity comparisons do
    not. Perplexity is reported for completeness and should not be quoted.

Usage::

    python CALM/suite/t2r_qknorm.py --time-only
    python CALM/suite/t2r_qknorm.py --seeds 0,1,2 --out CALM/suite/t2r_results.json
"""

from __future__ import annotations

import argparse
import json
import math
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
from hybrid import HybridDecoder  # noqa: E402
from t1_head_geometry import perplexity, train  # noqa: E402
from t2_criticality import measure  # noqa: E402

ARMS = [(qk, beta) for qk in (False, True)
        for beta in ("fixed", "learned")]


def name_of(qk_norm: bool, beta_mode: str) -> str:
    return f"{'qknorm' if qk_norm else 'raw':>6s}/{beta_mode}"


def build(vocab, options, qk_norm, beta_mode, seed):
    torch.manual_seed(seed)
    heads = max(options.dim // options.head_dim, 1)
    return HybridDecoder(vocab, dim=options.dim, layers=options.layers,
                         heads=heads, head_dim=options.head_dim,
                         kv_latent=options.kv_latent, ffn="dense",
                         beta_mode=beta_mode, qk_norm=qk_norm,
                         max_seq_len=options.seq_len)


def assert_paired(vocab, options, seed) -> int:
    """Every arm must start from the same weights, or no contrast is paired."""
    reference = build(vocab, options, *ARMS[0], seed).state_dict()
    shared = 0
    for qk_norm, beta_mode in ARMS[1:]:
        other = build(vocab, options, qk_norm, beta_mode, seed).state_dict()
        extra = sorted(set(other) - set(reference))
        expected = ["log_beta"] if beta_mode == "learned" else []
        if [e.split(".")[-1] for e in extra] != expected * len(extra):
            raise SystemExit(f"{name_of(qk_norm, beta_mode)} adds unexpected "
                             f"tensors: {extra}")
        for key, value in reference.items():
            if not torch.equal(value, other[key]):
                raise SystemExit(f"PAIRING BROKEN: {key} differs between "
                                 f"{name_of(*ARMS[0])} and "
                                 f"{name_of(qk_norm, beta_mode)}")
        shared = len(reference)
    return shared


def paired_diffs(rows, key, beta_a="fixed", beta_b="learned"):
    """`key` for (learned - fixed) at each seed, per qk_norm setting."""
    out = {}
    for qk_norm in (False, True):
        pairs = []
        for seed in sorted({r["seed"] for r in rows}):
            def find(mode):
                return next((r for r in rows if r["seed"] == seed
                             and r["qk_norm"] == qk_norm
                             and r["beta_mode"] == mode), None)
            a, b = find(beta_a), find(beta_b)
            if a and b and key in a and key in b:
                pairs.append(b[key] - a[key])
        out[qk_norm] = pairs
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--kv-latent", type=int, default=48)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--vocab", type=int, default=16000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--stat-batches", type=int, default=4)
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

    print(f"T2-redux  wikitext2 BPE {vocab}, dim {options.dim}, "
          f"{options.layers} layers, head_dim {options.head_dim}, dense FFN")
    print(f"    {options.steps} steps, lr {options.lr:.0e}, seq {options.seq_len}, "
          f"seeds {seeds}")
    print(f"    1/sqrt(d) = {1 / math.sqrt(options.head_dim):.6f}; "
          f"qk_norm pins |q||k| to head_dim = {options.head_dim}")

    if options.time_only:
        for qk_norm, beta_mode in ARMS:
            model = build(vocab, options, qk_norm, beta_mode, 0).to(device)
            stream = stream_from(train_ids, options.batch, options.seq_len, 0)
            _, per_step = train(model, stream, 8, options.lr, device)
            print(f"  {name_of(qk_norm, beta_mode):16s} {per_step*1000:7.1f} "
                  f"ms/step -> {options.steps} steps = "
                  f"{per_step*options.steps/60:5.1f} min")
        return

    shared = assert_paired(vocab, options, seeds[0])
    print(f"    pairing verified: {shared} shared tensors identical across all "
          f"four arms\n")

    rows: List[Dict] = []
    print(f"{'arm':>16s} {'seed':>5s} {'perplexity':>11s} {'beta_x':>7s} "
          f"{'qk_gain':>8s} {'part_frac':>10s} {'entropy':>8s}")
    for qk_norm, beta_mode in ARMS:
        for seed in seeds:
            model = build(vocab, options, qk_norm, beta_mode, seed).to(device)
            stream = stream_from(train_ids, options.batch, options.seq_len, 0)
            model, per_step = train(model, stream, options.steps, options.lr,
                                    device)
            ppl = perplexity(model, eval_batches)
            stats = measure(model, eval_batches[:options.stat_batches])
            row = {"qk_norm": qk_norm, "beta_mode": beta_mode, "seed": seed,
                   "perplexity": ppl, "ms_per_step": per_step * 1000, **stats}
            rows.append(row)
            print(f"{name_of(qk_norm, beta_mode):>16s} {seed:5d} {ppl:11.2f} "
                  f"{stats.get('beta_ratio', float('nan')):7.3f} "
                  f"{stats.get('qk_gain', float('nan')):8.3f} "
                  f"{stats.get('participation_frac', float('nan')):10.4f} "
                  f"{stats.get('entropy_norm', float('nan')):8.4f}", flush=True)
            if options.out:
                Path(options.out).parent.mkdir(parents=True, exist_ok=True)
                Path(options.out).write_text(json.dumps(
                    {"config": vars(options), "rows": rows}, indent=2,
                    default=lambda v: None))
    report(rows, options)


def mean_sd(values):
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], float("nan")
    return statistics.mean(values), statistics.stdev(values)


def report(rows, options):
    print("\n" + "=" * 78)
    print(f"{'arm':>16s} {'n':>3s} {'mean ppl':>10s} {'sd':>7s} "
          f"{'beta_x':>8s} {'qk_gain':>9s} {'part_frac':>10s} {'entropy':>8s}")
    for qk_norm, beta_mode in ARMS:
        group = [r for r in rows if r["qk_norm"] == qk_norm
                 and r["beta_mode"] == beta_mode]
        if not group:
            continue
        ppl, sd = mean_sd([r["perplexity"] for r in group])
        print(f"{name_of(qk_norm, beta_mode):>16s} {len(group):3d} {ppl:10.2f} "
              f"{sd:7.2f} "
              f"{mean_sd([r.get('beta_ratio', float('nan')) for r in group])[0]:8.3f} "
              f"{mean_sd([r.get('qk_gain', float('nan')) for r in group])[0]:9.3f} "
              f"{mean_sd([r.get('participation_frac', float('nan')) for r in group])[0]:10.4f} "
              f"{mean_sd([r.get('entropy_norm', float('nan')) for r in group])[0]:8.4f}")

    print("\nP1  does closing the magnitude route make beta move?")
    for qk_norm in (False, True):
        ratios = [r["beta_ratio"] for r in rows if r["qk_norm"] == qk_norm
                  and r["beta_mode"] == "learned" and "beta_ratio" in r]
        mean, sd = mean_sd(ratios)
        print(f"    qk_norm {'on ' if qk_norm else 'off'}: beta_ratio "
              f"{mean:.4f} +- {sd:.4f}   {[round(v, 4) for v in ratios]}")

    print("\nP2  does the q/k gain move to cancel beta?  (learned - fixed, paired)")
    diffs = paired_diffs(rows, "qk_gain")
    for qk_norm in (False, True):
        pairs = diffs[qk_norm]
        mean, sd = mean_sd(pairs)
        agree = sum(1 for v in pairs if v > 0)
        print(f"    qk_norm {'on ' if qk_norm else 'off'}: d(qk_gain) "
              f"{mean:+.4f} +- {sd:.4f}, {agree}/{len(pairs)} positive   "
              f"{[round(v, 4) for v in pairs]}")
    print("    (prediction: positive and consistent with qk_norm off; "
          "exactly 0 with it on)")

    print("\nP3  operating point, against T2's part_frac 0.20 +- 0.02 / "
          "entropy 0.41 +- 0.03")
    for qk_norm, beta_mode in ARMS:
        group = [r for r in rows if r["qk_norm"] == qk_norm
                 and r["beta_mode"] == beta_mode]
        if group:
            print(f"    {name_of(qk_norm, beta_mode):>16s} "
                  f"part_frac {mean_sd([r['participation_frac'] for r in group])[0]:.4f} "
                  f"entropy {mean_sd([r['entropy_norm'] for r in group])[0]:.4f}")

    print("\nP4  perplexity, paired (learned - fixed) -- expected to be noise")
    for qk_norm, pairs in paired_diffs(rows, "perplexity").items():
        mean, sd = mean_sd(pairs)
        print(f"    qk_norm {'on ' if qk_norm else 'off'}: d(ppl) {mean:+.2f} "
              f"+- {sd:.2f}   {[round(v, 2) for v in pairs]}")


if __name__ == "__main__":
    main()
