#!/usr/bin/env python3
"""Triage: the three gates from `../ALLOCATION.md` that fit in a 2-hour budget.

The full allocation suite is ~63 CPU-hours. In two hours no perplexity
comparison in it resolves -- T2 established that seed sd is 1.5-5.0 perplexity
at this scale, and T3 that even a *paired* design needs ~6 seeds to see 1%. So
this module deliberately does not attempt one. It runs the three checks that
are **gates or measurements** rather than comparisons, chosen because each can
kill an idea outright at a cost the deadline allows, and because T2's
methodological finding says instrumentation of internal state is roughly an
order of magnitude more sensitive than the loss at equal compute.

What is bought, and what is given up:

* **G1 -- T12 Gate A. Adjacent-layer redundancy.** Fit a least-squares map from
  block L's residual stream to block L+1's and report R^2. Predictive coding's
  premise is that layers are predictable from each other, so that forwarding
  only the unexplained part is a better code. If R^2 is low, there is no
  redundancy to remove and idea 7 (residual-only propagation) is dead before
  any architecture is written. Cost: 2 short training runs.

* **G2 -- T7 P7.2. Learned router vs random router.** Identical MoE model, same
  seed, same data order; in the control the router is frozen at its random
  initialisation. This is the control T2's MoE arm lacked, and without it a
  null cannot be diagnosed: "routing helps" and "having more experts helps"
  are different claims. Perplexity here is UNDERPOWERED and is reported with
  that label; the load-bearing numbers are the order parameters and the
  token/expert mutual information, which reproduced to three significant
  figures in T2 while perplexity moved 3.6%.

* **G3 -- T9 P9.1. The surprisal oracle bracket.** Eval-only. For every held-out
  token compute total surprisal under the full model, and *reducible*
  surprisal = surprisal at truncated depth minus surprisal at full depth --
  how much the last layers actually improved that token. Routing on total
  surprisal spends compute on tokens that are merely unpredictable (aleatoric);
  routing on reducible surprisal spends it where compute helps (epistemic).
  Their correlation, and the overlap of the token sets they would select, is
  the size of that confound measured rather than argued.

Given up, and stated so it is not later mistaken for having been tested: the
arithmetic-coding round trip (T6) needs a coder we do not have; MoD/MoR (T8)
needs a new block; the topology sweep (T11) needs mask generators and synthetic
tasks. None of those fit, and running a cut-down version of any of them would
produce a number without its gate.

**Predictions, registered before the run:**

G1. Mean adjacent-layer R^2 > 0.7 in the middle blocks, lower at the first and
    last. Basis: adjacent-layer representational similarity is a robust finding
    in deep models. **Kill condition: mean R^2 < 0.5 stops T12 permanently.**

G2. The learned router beats the frozen random router on perplexity, but by
    less than the seed sd -- i.e. not resolvable here, which is the honest
    expectation at 6 seeds and 500 steps and is why the MI is measured.
    Token/expert mutual information is > 0 for the learned router and ~0 for
    the frozen one. **If MI is ~0 for both, T7's premise is in trouble** and
    T8/T9 need re-scoping before they are run.

G3. Total and reducible surprisal correlate positively but weakly (Spearman
    rho in 0.2-0.5), and the top-50% token sets they select overlap well below
    1.0. That gap is the epistemic/aleatoric cost. **If rho > 0.8 and overlap
    > 0.85, the confound is quantitatively unimportant** and deployable
    surprisal routing becomes a live candidate rather than a doubtful one.

This module reports gates. It does not report findings about language models.

Usage::

    python CALM/suite/triage.py --seeds 0,1,2,3,4,5
    python CALM/suite/triage.py --smoke
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
from hybrid import MoE, HybridDecoder  # noqa: E402
from t1_head_geometry import perplexity, train  # noqa: E402
from t2_criticality import measure  # noqa: E402

T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}


def build(vocab, o, ffn, seed):
    torch.manual_seed(seed)
    return HybridDecoder(vocab, dim=o.dim, layers=o.layers,
                         heads=max(o.dim // o.head_dim, 1), head_dim=o.head_dim,
                         kv_latent=o.kv_latent, ffn=ffn, max_seq_len=o.seq_len,
                         n_experts=o.n_experts, top_k=o.top_k)


def freeze_routers(model) -> int:
    """The G2 control: routers stay at their random initialisation.

    Freezing rather than deleting keeps the arm bit-identical to the learned
    arm everywhere else, so the comparison is paired on every other tensor.
    """
    frozen = 0
    for module in model.modules():
        if isinstance(module, MoE):
            module.router.weight.requires_grad_(False)
            frozen += 1
    return frozen


# --- G1: adjacent-layer redundancy -------------------------------------------

@torch.no_grad()
def block_activations(model, tokens) -> List[torch.Tensor]:
    """Residual stream entering each block, plus the final stream."""
    x = model.embed(tokens)
    states = [x]
    for block in model.blocks:
        x = block(x)
        states.append(x)
    return states


@torch.no_grad()
def redundancy(model, batches) -> Dict[str, float]:
    """R^2 of a least-squares map from each block's input to the next block's.

    A bias column is included, and the fit is on the same data it is scored on,
    which makes this an UPPER bound on predictability -- deliberately, since the
    gate is a kill condition: an upper bound that fails to clear 0.5 is decisive
    in a way a cross-validated estimate that fails would not be.
    """
    states = [torch.cat(s, dim=0) for s in
              zip(*[block_activations(model, b[:, :-1]) for b in batches])]
    scores = []
    for lower, upper in zip(states[:-1], states[1:]):
        a = lower.reshape(-1, lower.size(-1)).double()
        b = upper.reshape(-1, upper.size(-1)).double()
        a = torch.cat([a, torch.ones(a.size(0), 1, dtype=a.dtype)], dim=1)
        weights = torch.linalg.lstsq(a, b).solution
        residual = b - a @ weights
        total = b - b.mean(0, keepdim=True)
        scores.append(1.0 - (residual.pow(2).sum() / total.pow(2).sum()).item())
    return {"per_pair": scores, "mean_r2": statistics.mean(scores),
            "middle_r2": statistics.mean(scores[1:-1]) if len(scores) > 2
            else statistics.mean(scores)}


# --- G2: does the router route? ----------------------------------------------

@torch.no_grad()
def expert_mi(model, batches, vocab) -> Dict[str, float]:
    """Mutual information between token identity and top-1 expert, in bits.

    Zero means the router's choice carries no information about the token --
    which is what a frozen random projection of a (normalised) hidden state
    should approach, and what a collapsed learned router would also show. The
    two are distinguished by load balance, reported alongside.
    """
    joint: Dict[tuple, int] = {}
    n_experts = 0
    for tokens in batches:
        flat = tokens[:, :-1].reshape(-1)
        picks = []
        x = model.embed(tokens[:, :-1])
        for block in model.blocks:
            if isinstance(block.ffn, MoE):
                n_experts = block.ffn.n_experts
                hidden = block.norm2(x + block.attn(block.norm1(x)))
                choice = block.ffn.router(hidden.reshape(-1, hidden.size(-1)))
                picks.append(choice.argmax(-1))
            x = block(x)
        if not picks:
            return {"mi_bits": float("nan"), "mi_frac": float("nan")}
        for chosen in picks:
            for t, e in zip(flat.tolist(), chosen.tolist()):
                joint[(t, e)] = joint.get((t, e), 0) + 1
    total = sum(joint.values())
    p_token: Dict[int, float] = {}
    p_expert: Dict[int, float] = {}
    for (t, e), c in joint.items():
        p_token[t] = p_token.get(t, 0.0) + c / total
        p_expert[e] = p_expert.get(e, 0.0) + c / total
    mi = sum((c / total) * math.log2((c / total) / (p_token[t] * p_expert[e]))
             for (t, e), c in joint.items())
    ceiling = math.log2(max(n_experts, 2))
    return {"mi_bits": mi, "mi_frac": mi / ceiling}


# --- G3: the surprisal oracle bracket ----------------------------------------

@torch.no_grad()
def truncated_logits(model, tokens, blocks: int):
    x = model.embed(tokens)
    for block in model.blocks[:blocks]:
        x = block(x)
    return model.head(model.norm(x))


@torch.no_grad()
def oracle_bracket(model, batches, keep: int) -> Dict[str, float]:
    """Per-token total surprisal vs the part of it that extra depth removes.

    `total` is what an entropy/surprisal router would see. `reducible` is
    surprisal at `keep` blocks minus surprisal at full depth: the measured
    improvement from spending the remaining compute on this token, which is
    what a router *should* see and cannot compute in advance.
    """
    total, reducible = [], []
    for tokens in batches:
        target = tokens[:, 1:].reshape(-1)
        full = torch.nn.functional.cross_entropy(
            model.logits(tokens[:, :-1]).float().reshape(-1, model.head.out_features),
            target, reduction="none")
        short = torch.nn.functional.cross_entropy(
            truncated_logits(model, tokens[:, :-1], keep).float()
            .reshape(-1, model.head.out_features), target, reduction="none")
        total.append(full)
        reducible.append(short - full)
    total = torch.cat(total).double()
    reducible = torch.cat(reducible).double()

    def spearman(a, b):
        ra = a.argsort().argsort().double()
        rb = b.argsort().argsort().double()
        ra = ra - ra.mean()
        rb = rb - rb.mean()
        return (ra @ rb / (ra.norm() * rb.norm())).item()

    half = total.numel() // 2
    top_total = set(total.topk(half).indices.tolist())
    top_reducible = set(reducible.topk(half).indices.tolist())
    overlap = len(top_total & top_reducible) / max(len(top_total), 1)
    return {
        "spearman": spearman(total, reducible),
        "top_half_overlap": overlap,
        "mean_total": total.mean().item(),
        "mean_reducible": reducible.mean().item(),
        # What fraction of a token's surprisal the last blocks actually remove.
        # Small means most surprisal is aleatoric at this scale and no amount of
        # routing on it can help.
        "reducible_share": (reducible.mean() / total.mean()).item(),
    }


def paired_t(diffs):
    if len(diffs) < 2:
        return float("nan"), float("nan")
    mean = statistics.mean(diffs)
    sd = statistics.stdev(diffs)
    t = mean / (sd / math.sqrt(len(diffs))) if sd > 0 else float("inf")
    return t, T_CRIT.get(len(diffs) - 1, 2.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--dim", type=int, default=192)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=32)
    p.add_argument("--kv-latent", type=int, default=48)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--vocab", type=int, default=16000)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--n-experts", type=int, default=4)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--seeds", default="0,1,2,3,4,5")
    p.add_argument("--g1-seeds", default="0,1")
    p.add_argument("--eval-batches", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--out", default="CALM/suite/triage_results.json")
    o = p.parse_args()
    if o.smoke:
        o.steps, o.seeds, o.g1_seeds, o.eval_batches = 8, "0", "0", 2

    device = torch.device(o.device)
    tokenizer = train_or_load("wikitext2", o.vocab)
    train_ids = encode_split(tokenizer, "wikitext2", "train")
    valid_ids = encode_split(tokenizer, "wikitext2", "valid")
    vocab = tokenizer.get_vocab_size()
    eval_batches = [b.to(device) for b in batches_from(
        valid_ids, o.batch, o.seq_len, o.eval_batches, seed=1)]
    seeds = [int(v) for v in o.seeds.split(",")]
    g1_seeds = [int(v) for v in o.g1_seeds.split(",")]
    started = time.time()

    print("TRIAGE  the three gates from ALLOCATION.md that fit a 2-hour budget")
    print(f"    wikitext2 BPE {vocab}, dim {o.dim}, {o.layers} layers, "
          f"head_dim {o.head_dim}, lr {o.lr:.1e}, {o.steps} steps")
    print("    G1 layer redundancy (kill T12 if mean R2 < 0.5)")
    print("    G2 learned vs FROZEN-RANDOM router (the control T2 lacked)")
    print("    G3 total vs reducible surprisal (the epistemic/aleatoric gap)\n")

    results: Dict[str, object] = {"config": vars(o), "vocab": vocab}

    # --- G1 + G3 share their models ------------------------------------------
    print("G1/G3  dense model, redundancy probe and oracle bracket")
    g1_rows, g3_rows = [], []
    for seed in g1_seeds:
        model = build(vocab, o, "dense", seed).to(device)
        # R2 at initialisation, because an untrained block is close to the
        # identity and would pass the gate trivially. The gate is about
        # redundancy the model LEARNED, so the trained figure is only
        # meaningful beside this one.
        init_r2 = redundancy(model.eval(), eval_batches[:4])["mean_r2"]
        stream = stream_from(train_ids, o.batch, o.seq_len, 0)
        model, per_step = train(model, stream, o.steps, o.lr, device)
        ppl = perplexity(model, eval_batches)
        red = redundancy(model, eval_batches[:4])
        bracket = oracle_bracket(model, eval_batches[:8],
                                 keep=max(o.layers - 1, 1))
        g1_rows.append({"seed": seed, "perplexity": ppl, "init_r2": init_r2,
                        "ms_per_step": per_step * 1000, **red})
        g3_rows.append({"seed": seed, **bracket})
        print(f"    seed {seed}: ppl {ppl:8.2f}  mean_R2 {red['mean_r2']:.4f} "
              f"(init {init_r2:.4f})  middle_R2 {red['middle_r2']:.4f}  "
              f"pairs {[round(v, 3) for v in red['per_pair']]}")
        print(f"             rho(total, reducible) {bracket['spearman']:+.4f}  "
              f"top-half overlap {bracket['top_half_overlap']:.4f}  "
              f"reducible share {bracket['reducible_share']:.4f}")
    results["g1"] = g1_rows
    results["g3"] = g3_rows

    # --- G2 -------------------------------------------------------------------
    print("\nG2     MoE, learned router vs router frozen at initialisation")
    g2_rows = []
    for seed in seeds:
        for arm in ("learned", "frozen"):
            model = build(vocab, o, "moe", seed).to(device)
            if arm == "frozen":
                freeze_routers(model)
            stream = stream_from(train_ids, o.batch, o.seq_len, 0)
            model, per_step = train(model, stream, o.steps, o.lr, device)
            ppl = perplexity(model, eval_batches)
            stats = measure(model, eval_batches[:4])
            mi = expert_mi(model, eval_batches[:4], vocab)
            g2_rows.append({"seed": seed, "arm": arm, "perplexity": ppl,
                            "ms_per_step": per_step * 1000, **stats, **mi})
            print(f"    seed {seed} {arm:7s}: ppl {ppl:8.2f}  "
                  f"MI {mi['mi_bits']:.4f} bits ({mi['mi_frac']:.3f} of ceiling)"
                  f"  load_bal {stats.get('load_balance', float('nan')):.4f}  "
                  f"router_ent {stats.get('router_entropy_norm', float('nan')):.4f}")
    results["g2"] = g2_rows

    # --- verdicts -------------------------------------------------------------
    print("\n" + "=" * 72)
    mean_r2 = statistics.mean(r["mean_r2"] for r in g1_rows)
    mid_r2 = statistics.mean(r["middle_r2"] for r in g1_rows)
    init_r2 = statistics.mean(r["init_r2"] for r in g1_rows)
    print(f"G1  mean adjacent-layer R2 {mean_r2:.4f} (middle {mid_r2:.4f}), "
          f"at init {init_r2:.4f}")
    if mean_r2 >= 0.5 and init_r2 >= mean_r2:
        print("    NOTE: training REDUCED redundancy. The gate passes on the "
              "level but the\n    direction says depth is being used, not "
              "wasted -- read stage B's prediction again.")
    verdict = ("PASS -- layers are redundant, T12 stage B is worth building"
               if mean_r2 >= 0.5 else
               "KILL -- no redundancy to remove; T12 stops here")
    print(f"    {verdict}")

    by = {arm: {r["seed"]: r for r in g2_rows if r["arm"] == arm}
          for arm in ("learned", "frozen")}
    shared = sorted(set(by["learned"]) & set(by["frozen"]))
    diffs = [by["frozen"][s]["perplexity"] - by["learned"][s]["perplexity"]
             for s in shared]
    t, crit = paired_t(diffs)
    mi_learned = statistics.mean(by["learned"][s]["mi_bits"] for s in shared)
    mi_frozen = statistics.mean(by["frozen"][s]["mi_bits"] for s in shared)
    print(f"\nG2  paired frozen-minus-learned perplexity: "
          f"mean {statistics.mean(diffs):+.2f}, {sum(d > 0 for d in diffs)}/"
          f"{len(diffs)} positive, t {t:.2f} (crit {crit:.3f})")
    print(f"    token/expert MI: learned {mi_learned:.4f} bits, "
          f"frozen {mi_frozen:.4f} bits")
    verdict = ("PASS -- the learned router carries token information the "
               "random one does not"
               if mi_learned > 2 * max(mi_frozen, 1e-9) else
               "FLAG -- routing carries no more information than a random "
               "projection; T7 P7.2 is in doubt and T8/T9 need re-scoping")
    print(f"    {verdict}")

    rho = statistics.mean(r["spearman"] for r in g3_rows)
    ov = statistics.mean(r["top_half_overlap"] for r in g3_rows)
    share = statistics.mean(r["reducible_share"] for r in g3_rows)
    print(f"\nG3  rho(total, reducible) {rho:+.4f}, top-half overlap {ov:.4f}, "
          f"reducible share of surprisal {share:.4f}")
    verdict = ("the confound is small -- surprisal routing is a live candidate"
               if rho > 0.8 and ov > 0.85 else
               "the confound is real -- a router on total surprisal would "
               "spend compute on tokens extra depth does not help")
    print(f"    {verdict}")
    print("=" * 72)
    print(f"\nwall clock {(time.time() - started) / 60:.1f} min")
    print("Underpowered by construction: 500 steps is 0.2 epochs and 6 seeds "
          "against a seed sd of\n1.5-5.0 perplexity. The perplexity column "
          "here is a diagnostic, not a result.")

    if o.out:
        Path(o.out).write_text(json.dumps(results, indent=1))
        print(f"wrote {o.out}")


if __name__ == "__main__":
    main()
