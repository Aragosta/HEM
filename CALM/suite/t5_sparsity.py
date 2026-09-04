#!/usr/bin/env python3
"""T5: learned sparsity against MHA and MoE MHA -- and does the MECHANISM matter?

See `../SPARSITY.md` for the full argument. In short:

Three ways to normalise attention scores, differing in what happens to a key
the model has decided to ignore:

* **softmax** -- full support. Every key keeps a strictly positive weight, so
  a temperature can concentrate attention but never remove a key. This is the
  baseline, and it is what T2's learned `beta` was confined to.
* **alpha-entmax** -- `[(alpha-1)z - tau]_+^{1/(alpha-1)}`, learnable `alpha`
  per head. The `[.]_+` is a ReLU, so it produces EXACT zeros -- and its
  Jacobian is exactly zero there too. A zeroed key receives no gradient and
  cannot be learned back. Measured: 7 of 7 zeroed keys had gradient 0, and 200
  SGD steps of direct pressure failed to revive one. Hard sparsity is an
  absorbing state. **This arm is the negative control, not the candidate.**
* **sigmoid** -- independent per-key gates `sigmoid(q.k + b)`, no simplex, no
  competition, no threshold. Weights approach zero without reaching it, so the
  gradient never dies (measured min |grad| 1.5e-01). Sparsity is soft and
  recoverable. Costs 1.00x softmax (0.96x with MoE): free.

**Allocation.** softmax and sigmoid get 6 seeds; entmax gets 3. The
softmax/sigmoid contrast is the practical question and needs the power; entmax
is bought only to answer whether the dead-gradient defect costs anything large.
entmax is 2.0-2.3x the cost of the other two, so equal seeds would spend half
the budget on the control.

**Pairing.** Within a fixed FFN type, all three arms share every weight tensor
bit-identically at a given seed -- the extra parameters are per-head scalars
(`alpha_logit`, `gate_bias`) that do not perturb the shared weights. The data
stream seed is fixed and there is no dropout. So the normaliser contrast is
paired, as in T3, where pairing made a 1% effect measurable that T2's unpaired
design would have needed ~79 seeds to see. `assert_paired` enforces this.

**Predictions, registered before the run:**

P0. sigmoid beats entmax. If they land together, the dead-gradient defect is
    correct in mechanism but not load-bearing at this scale -- a useful
    negative result and the reason the control is worth its cost.
P1. entmax is at parity with softmax, not a large win. The source paper's own
    result is +0.11 BLEU on WMT14. A large gain means a bug.
P2. Learned sparsity moves toward DENSITY. alpha falls from 1.5 toward 1.
    Basis: Correia et al. find decoder self-attention prefers denser attention
    than encoder; T2 found learned beta went to 0.934x (flatter); a 40-step
    probe here gave 1.500 -> 1.490. Two mechanisms, two codebases, same
    direction in autoregressive attention.
P3. No interaction with MoE routing. Attention sparsity reduces the
    token-token graph, MoE routing reduces the token-expert graph; there is no
    mechanism connecting them. An interaction would be a real surprise.
P4. **The honest null.** Sparsity's claimed benefit is preventing attention
    dispersion at long context (arXiv:2506.16640). At seq_len 128 there is
    little dispersion to prevent, so a null here is WEAK evidence and must be
    reported as "not at this context length", never as "sparsity does not
    help". If P0/P1 are null, the informative follow-up is the length axis,
    not more seeds.

Usage::

    python CALM/suite/t5_sparsity.py --seeds 0,1,2,3,4,5 --entmax-seeds 0,1,2
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
from hybrid import HybridDecoder  # noqa: E402
from t1_head_geometry import perplexity, train  # noqa: E402
from t2_criticality import measure  # noqa: E402
from t3_paired_geometry import sign_test_p  # noqa: E402

T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
          7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179}


def build(vocab, options, ffn, normalizer, seed):
    torch.manual_seed(seed)
    return HybridDecoder(
        vocab, dim=options.dim, layers=options.layers,
        heads=max(options.dim // options.head_dim, 1),
        head_dim=options.head_dim, kv_latent=options.kv_latent, ffn=ffn,
        normalizer=normalizer, max_seq_len=options.seq_len)


def assert_paired(vocab, options, ffn):
    """The paired statistic is only valid if the shared weights are identical.

    Checked rather than assumed: if a future change adds a weight tensor on one
    branch, pairing breaks silently and the paired difference becomes wrong
    rather than merely noisy.
    """
    base = dict(build(vocab, options, ffn, "softmax", 999).named_parameters())
    for normalizer in ("entmax", "sigmoid"):
        other = dict(build(vocab, options, ffn, normalizer, 999).named_parameters())
        shared = [k for k in base if k in other]
        if not all(torch.equal(base[k], other[k]) for k in shared):
            raise SystemExit(
                f"PAIRING BROKEN: {ffn}/{normalizer} does not share "
                f"initialisation with {ffn}/softmax; a paired comparison would "
                f"be invalid.")
        extra = sorted(set(other) - set(base))
        print(f"    {ffn:5s} {normalizer:8s}: {len(shared)} shared tensors "
              f"identical, extra {extra if extra else 'none'}")


def paired_report(rows, ffn, arm, baseline="softmax"):
    """Per-seed differences of `arm` against `baseline`, at fixed ffn."""
    def get(normalizer):
        return {r["seed"]: r["perplexity"] for r in rows
                if r["ffn"] == ffn and r["normalizer"] == normalizer}
    a, b = get(baseline), get(arm)
    seeds = sorted(set(a) & set(b))
    return [(s, b[s] - a[s]) for s in seeds]


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
    parser.add_argument("--seeds", default="0,1,2,3,4,5")
    parser.add_argument("--entmax-seeds", default="0,1,2",
                        help="entmax is 2.0-2.3x the cost and is the control")
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
    seeds = [int(v) for v in options.seeds.split(",")]
    entmax_seeds = [int(v) for v in options.entmax_seeds.split(",")]

    print("T5  learned sparsity: does the MECHANISM of sparsification matter?")
    print(f"    wikitext2 BPE {vocab}, dim {options.dim}, {options.layers} layers, "
          f"head_dim {options.head_dim}, lr {options.lr:.1e}, {options.steps} steps")
    print(f"    softmax/sigmoid seeds {seeds}; entmax seeds {entmax_seeds} "
          f"(control, 2.0-2.3x cost)")
    print("    P0 sigmoid > entmax; P1 entmax ~ softmax; P2 sparsity moves "
          "toward DENSITY;\n    P3 no MoE interaction; P4 a null is weak "
          "evidence at seq_len 128.\n")
    print("    pairing check:")
    for ffn in ("dense", "moe"):
        assert_paired(vocab, options, ffn)
    print()

    plan = []
    for ffn in ("dense", "moe"):
        for normalizer in ("softmax", "sigmoid", "entmax"):
            for seed in (entmax_seeds if normalizer == "entmax" else seeds):
                plan.append((ffn, normalizer, seed))

    rows: List[Dict] = []
    print(f"{'ffn':>6s} {'attn':>8s} {'seed':>5s} {'perplexity':>11s} "
          f"{'part_frac':>10s} {'sparsity':>9s} {'alpha':>7s} {'load_bal':>9s}")
    for ffn, normalizer, seed in plan:
        model = build(vocab, options, ffn, normalizer, seed).to(device)
        stream = stream_from(train_ids, options.batch, options.seq_len, 0)
        model, per_step = train(model, stream, options.steps, options.lr, device)
        ppl = perplexity(model, eval_batches)
        stats = measure(model, eval_batches[:4])
        # "sparsity" means exact zeros for entmax and near-zero gates for
        # sigmoid; they are not the same quantity and are labelled as such in
        # the JSON, but both answer "how many keys did this head drop?"
        sparsity = stats.get("zero_frac", stats.get("near_zero_frac", float("nan")))
        if normalizer == "sigmoid":
            sparsity = stats.get("near_zero_frac", float("nan"))
        row = {"ffn": ffn, "normalizer": normalizer, "seed": seed,
               "perplexity": ppl, "ms_per_step": per_step * 1000,
               "sparsity": sparsity, **stats}
        rows.append(row)
        print(f"{ffn:>6s} {normalizer:>8s} {seed:5d} {ppl:11.2f} "
              f"{stats.get('participation_frac', float('nan')):10.4f} "
              f"{sparsity:9.4f} {stats.get('alpha', float('nan')):7.4f} "
              f"{stats.get('load_balance', float('nan')):9.4f}", flush=True)
        if options.out:
            Path(options.out).parent.mkdir(parents=True, exist_ok=True)
            Path(options.out).write_text(json.dumps(
                {"config": vars(options), "rows": rows}, indent=2))
    report(rows)


def report(rows):
    def mean_of(ffn, normalizer, key):
        vals = [r[key] for r in rows if r["ffn"] == ffn
                and r["normalizer"] == normalizer and r.get(key) == r.get(key)]
        return statistics.mean(vals) if vals else float("nan")

    print("\n" + "=" * 76)
    print(f"{'arm':>16s} {'n':>3s} {'mean ppl':>10s} {'sd':>7s} "
          f"{'part_frac':>10s} {'sparsity':>9s}")
    for ffn in ("dense", "moe"):
        for normalizer in ("softmax", "sigmoid", "entmax"):
            group = [r["perplexity"] for r in rows
                     if r["ffn"] == ffn and r["normalizer"] == normalizer]
            if not group:
                continue
            sd = statistics.stdev(group) if len(group) > 1 else 0.0
            print(f"{ffn + '/' + normalizer:>16s} {len(group):3d} "
                  f"{statistics.mean(group):10.2f} {sd:7.2f} "
                  f"{mean_of(ffn, normalizer, 'participation_frac'):10.4f} "
                  f"{mean_of(ffn, normalizer, 'sparsity'):9.4f}")

    print("\nPAIRED contrasts against softmax  (negative = the arm is BETTER)")
    print(f"{'contrast':>22s} {'n':>3s} {'mean':>8s} {'sd':>7s} {'t':>7s} "
          f"{'crit':>6s} {'signs':>7s}")
    for ffn in ("dense", "moe"):
        for arm in ("sigmoid", "entmax"):
            diffs = [d for _, d in paired_report(rows, ffn, arm)]
            if len(diffs) < 2:
                continue
            m, sd = statistics.mean(diffs), statistics.stdev(diffs)
            sem = sd / len(diffs) ** 0.5
            t = m / sem if sem else float("nan")
            crit = T_CRIT.get(len(diffs) - 1, 1.96)
            neg = sum(1 for d in diffs if d < 0)
            print(f"{ffn + '/' + arm + ' vs softmax':>22s} {len(diffs):3d} "
                  f"{m:+8.2f} {sd:7.2f} {t:+7.2f} {crit:6.3f} "
                  f"{neg:3d}/{len(diffs):<3d}")

    print("\nP0  sigmoid vs entmax, paired on the seeds both ran")
    for ffn in ("dense", "moe"):
        sig = dict(paired_report(rows, ffn, "sigmoid"))
        ent = dict(paired_report(rows, ffn, "entmax"))
        shared = sorted(set(sig) & set(ent))
        if len(shared) < 2:
            continue
        diffs = [sig[s] - ent[s] for s in shared]
        m, sd = statistics.mean(diffs), statistics.stdev(diffs)
        sem = sd / len(diffs) ** 0.5
        t = m / sem if sem else float("nan")
        crit = T_CRIT.get(len(diffs) - 1, 1.96)
        verdict = ("sigmoid better" if m < 0 else "entmax better")
        resolved = "RESOLVED" if abs(t) > crit else "not resolved"
        print(f"    {ffn:5s} n={len(diffs)}  mean {m:+.2f}  t={t:+.2f} "
              f"(crit {crit:.3f})  -> {verdict}, {resolved}")

    print("\nP2  direction of learned sparsity")
    for ffn in ("dense", "moe"):
        a = mean_of(ffn, "entmax", "alpha")
        if a == a:
            print(f"    {ffn:5s} entmax alpha {a:.4f} from 1.5000 init -> "
                  f"{'DENSER' if a < 1.5 else 'SPARSER'} (P2 predicts denser)")

    print("\n  Reminder (P4): at seq_len 128 there is little attention "
          "dispersion to\n  prevent, so a null here is weak evidence. It means "
          "'not at this context\n  length', not 'learned sparsity does not "
          "help'. The follow-up is the\n  length axis, not more seeds.")


if __name__ == "__main__":
    main()
