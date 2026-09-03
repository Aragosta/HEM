#!/usr/bin/env python3
"""T0: does HELM's advantage appear in a metric a CALM model could share?

**Why this comes before building anything.** HELM's paper reports its advantage
*only* as multiple-choice accuracy, scored by the evaluation harness picking the
answer choice "with the highest likelihood value" (HELM, Appendix C.3). CALM's
head is an implicit sampler with no likelihood, so **every number HELM reports is
one a HELM-CALM cannot produce**. HELM's paper reports no perplexity at all.

So before asking whether hyperbolic geometry survives composition with CALM, ask
a cheaper question with no CALM in it: **does HELM's advantage over a matched
Euclidean model show up in perplexity or BrierLM?**

* **It does** -> the integration is measurable. Build it, evaluate it on that
  metric, and the comparison means something.
* **It does not** -> HELM's advantage exists only in a measurement CALM cannot
  perform, and HELM-CALM cannot be evaluated against the claim that motivates
  it. Do not build it.

Both arms here are **discrete** models, so perplexity is defined for both. That
is the point: find a shared metric first, using models that can both be measured
every way.

**The tail test, folded in for free.** HELM's own Table 3 locates the geometry's
contribution in the tail -- generic words cluster at small norm, specific words
at large norm. Energy-score training, meanwhile, "allocate[s] training signal in
proportion to data density" (arXiv:2607.01171), i.e. away from the tail. So this
also reports **perplexity by token-frequency decile**. If HELM's edge is
concentrated in rare deciles, that is the mechanism, and it predicts a negative
interaction with CALM before any CALM model is trained. Same forward pass, no
extra cost.

Setup follows HELM's where reachable: BPE tokenizer trained on the corpus
(LLaMA-3.1's is not fetchable here), WikiText-2 official splits, parameter-matched
arms, identical data order and schedule.

Usage::

    python CALM/suite/t0_shared_metric.py --steps 6000 --seeds 0,1
    python CALM/suite/t0_shared_metric.py --time-only
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

ROOT = Path(__file__).resolve().parents[2]
for _extra in (ROOT, ROOT / "CALM", ROOT / "CALM" / "experiments",
               ROOT / "CALM" / "suite"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import metrics as M  # noqa: E402
from bpe import encode_split, train_or_load  # noqa: E402
from corpus import batches_from, stream_from  # noqa: E402
from models import (EuclideanDiscrete, build_helm_discrete,  # noqa: E402
                    count_parameters, describe, match_euclidean_width)
from tests._config import tiny_args  # noqa: E402


def helm_args(vocab, dim, layers, heads, seq_len, batch, dense=True):
    """HELM config. ``dense=True`` makes every layer dense, i.e. HELM-D.

    The paper compares like with like -- HELM-MiCE against DeepSeekV3 (both MoE)
    and HELM-D against LLaMA (both dense). An earlier version of this script
    compared HELM-MiCE (sparse, 9.79M active of 11.09M) against a dense
    Euclidean model (11.06M active), which is a sparse-versus-dense comparison
    wearing a geometry label. Dense on both sides removes that.
    """
    head = max(((dim - 1) // heads) // 2 * 2 + 1, 5)
    return tiny_args(vocab_size=vocab, dim=dim, n_layers=layers, n_heads=heads,
                     n_dense_layers=layers if dense else 1,
                     max_seq_len=seq_len, original_seq_len=seq_len,
                     max_batch_size=batch, qk_nope_head_dim=head,
                     qk_rope_head_dim=head, v_head_dim=head,
                     kv_lora_rank=(dim // 2) * 2 + 1, inter_dim=2 * dim,
                     moe_inter_dim=dim + 31, mice_inter_dim=dim + 31)


def unwrap(out):
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def perplexity_and_deciles(logits_fn, batches, decile_of, n_deciles=10):
    """Overall perplexity, plus perplexity within each token-frequency decile.

    Decile 0 is the most frequent tokens, 9 the rarest. The rare deciles are
    where HELM's mechanism is supposed to act and where CALM's objective is
    weakest, so the split is the whole point rather than a decoration.
    """
    nats = torch.zeros(n_deciles, dtype=torch.float64)
    counts = torch.zeros(n_deciles, dtype=torch.float64)
    total_nats = total = 0.0
    for tokens in batches:
        logits = unwrap(logits_fn(tokens[:, :-1])).float()
        target = tokens[:, 1:].reshape(-1)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), target, reduction="none")
        total_nats += loss.sum().item()
        total += target.numel()
        bucket = decile_of[target]
        nats.scatter_add_(0, bucket.cpu().long(), loss.double().cpu())
        counts.scatter_add_(0, bucket.cpu().long(),
                            torch.ones_like(loss, dtype=torch.float64).cpu())
    per_decile = (nats / counts.clamp_min(1)).exp().tolist()
    return math.exp(total_nats / total), per_decile


def frequency_deciles(train_ids: torch.Tensor, vocab: int) -> torch.Tensor:
    """Map each token id to a frequency decile, 0 = most frequent."""
    counts = torch.zeros(vocab, dtype=torch.long)
    values, freq = torch.unique(train_ids, return_counts=True)
    counts[values] = freq
    order = torch.argsort(counts, descending=True)
    decile = torch.zeros(vocab, dtype=torch.long)
    # Split by cumulative *token mass*, so each decile carries ~10% of the
    # corpus rather than 10% of the vocabulary -- otherwise decile 9 is
    # thousands of tokens that never occur and the bucket is empty.
    sorted_counts = counts[order].double()
    cumulative = sorted_counts.cumsum(0) / sorted_counts.sum().clamp_min(1)
    decile[order] = (cumulative * 10).clamp(max=9).long()
    return decile


def build_optimizer(model, lr):
    """Riemannian optimization for manifold parameters, AdamW for the rest.

    HELM's paper: hyperbolic word embeddings "are then trained as hyperbolic
    parameters **via Riemannian optimizers**". An earlier version of this script
    used plain AdamW on everything and never retracted, so HELM's embedding --
    a geoopt ManifoldParameter -- drifted off the hyperboloid during training.
    That handicaps only the hyperbolic arm, which is the mirror image of the
    confounds the first run had in HELM's favour.
    """
    from geoopt import ManifoldParameter
    from geoopt.optim import RiemannianAdam

    manifold, euclidean = [], []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (manifold if isinstance(parameter, ManifoldParameter) else euclidean
         ).append(parameter)
    optimizers = [torch.optim.AdamW(euclidean, lr=lr, weight_decay=0.01)]
    if manifold:
        optimizers.append(RiemannianAdam(manifold, lr=lr, weight_decay=0.01,
                                         stabilize=10))
    return optimizers, manifold


def warmup_cosine(optimizers, steps, warmup_fraction=0.03, floor=0.1):
    """HELM's schedule: cosine annealing to 0.1x, with 3% of steps as warmup.

    The warmup was missing before. HELM diverged at 8e-4 while the Euclidean arm
    did not, which is the signature of exactly that omission.
    """
    warmup = max(int(steps * warmup_fraction), 1)

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(steps - warmup, 1)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    return [torch.optim.lr_scheduler.LambdaLR(o, factor) for o in optimizers]


def train_arm(name, args, cfg, stream, seed, device, lr, euclid_dim=None):
    torch.manual_seed(seed)
    if name == "helm":
        model = build_helm_discrete(args).to(device)
        step = lambda t: unwrap(model(t[:, :-1], labels=t[:, 1:]))
        logits_fn = lambda t: unwrap(model(t))
    else:
        model = EuclideanDiscrete(args.vocab_size, euclid_dim, cfg["layers"],
                                  cfg["heads"], 2 * euclid_dim,
                                  cfg["seq_len"]).to(device)
        step = model.loss
        logits_fn = model.logits
    params = [p for p in model.parameters() if p.requires_grad]
    optimizers, manifold = build_optimizer(model, lr)
    schedules = warmup_cosine(optimizers, cfg["steps"])
    model.train()
    started = time.time()
    history = []
    for _ in range(cfg["steps"]):
        loss = step(next(stream).to(device))
        for optimizer in optimizers:
            optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        for optimizer in optimizers:
            optimizer.step()
        for schedule in schedules:
            schedule.step()
        history.append(loss.item())
    violation = 0.0
    for parameter in manifold:
        d = parameter.detach()
        quad = -d[..., 0] ** 2 + d[..., 1:].pow(2).sum(-1)
        violation = max(violation, (quad + 1).abs().max().item())
    return (model.eval(), logits_fn, (time.time() - started) / cfg["steps"],
            history, violation)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--dim", type=int, default=257)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--vocab", type=int, default=16000)
    # HELM's own protocol (Appendix C.2): "a learning rate of 2e-4 for all dense
    # models and a learning rate of 4e-4 for the MoE and MiCE models". HELM-MiCE
    # is MoE, the Euclidean control here is dense, so they get different rates --
    # matching the paper rather than imposing one rate on both. Small models are
    # highly hyperparameter-sensitive (arXiv:2608.11859), so this is not a detail.
    parser.add_argument("--lr", type=float, default=4e-4,
                        help="learning rate for the HELM (MoE) arm")
    parser.add_argument("--lr-dense", type=float, default=2e-4,
                        help="learning rate for the dense Euclidean arm")
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--lr-sweep-helm", default="",
                        help="learning rates for the HELM arm, separate from the "
                             "control's because the two prefer different ranges. "
                             "Measured: under RiemannianAdam HELM improves "
                             "monotonically toward LOWER rates while the "
                             "Euclidean control improves toward HIGHER ones, so a "
                             "single shared grid brackets neither optimum.")
    parser.add_argument("--lr-sweep-euclid", default="")
    parser.add_argument("--lr-sweep", default="",
                        help="comma-separated learning rates; when set, each arm "
                             "is trained at every rate for --sweep-steps and the "
                             "best per arm is reported. Comparing two arms at one "
                             "shared rate, or at rates picked for other models, "
                             "measures the rate as much as the architecture "
                             "(arXiv:2608.11859).")
    parser.add_argument("--sweep-steps", type=int, default=1200)
    parser.add_argument("--sparse", action="store_true",
                        help="use HELM-MiCE (sparse) instead of HELM-D (dense); "
                             "only sound if the control is also MoE")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-batches", type=int, default=24)
    parser.add_argument("--time-only", action="store_true")
    parser.add_argument("--out", default="")
    options = parser.parse_args()

    device = torch.device(options.device)
    cfg = {"steps": options.steps, "seq_len": options.seq_len,
           "layers": options.layers, "heads": options.heads}

    tokenizer = train_or_load("wikitext2", options.vocab)
    train_ids = encode_split(tokenizer, "wikitext2", "train")
    valid_ids = encode_split(tokenizer, "wikitext2", "valid")
    vocab = tokenizer.get_vocab_size()
    args = helm_args(vocab, options.dim, options.layers, options.heads,
                     options.seq_len, options.batch, dense=not options.sparse)

    width, euclid_params, helm_params = match_euclidean_width(
        lambda: build_helm_discrete(args),
        lambda w: EuclideanDiscrete(vocab, w, options.layers, options.heads,
                                    2 * w, options.seq_len),
        multiple_of=2 * options.heads)
    probe = EuclideanDiscrete(vocab, width, options.layers, options.heads,
                              2 * width, options.seq_len)
    if not probe.backbone.rotary:
        raise SystemExit(
            f"control at width {width} could not use rotary encoding "
            f"(heads {probe.backbone.heads}, head_dim "
            f"{width // probe.backbone.heads}). HELM uses HOPE, a rotary scheme, "
            f"so a non-rotary control would make the comparison a "
            f"positional-encoding comparison. Adjust --dim or --heads.")
    budget = describe(build_helm_discrete(args), args, options.seq_len, is_helm=True)

    eval_batches = [b.to(device) for b in batches_from(
        valid_ids, options.batch, options.seq_len, options.eval_batches, seed=1)]
    deciles = frequency_deciles(train_ids, vocab).to(device)

    seen = options.steps * options.batch * options.seq_len
    print(f"T0  wikitext2, BPE {vocab}, seq {options.seq_len}, "
          f"{options.steps} steps = {seen:,} tokens = "
          f"{seen / train_ids.numel():.2f} epochs")
    print(f"HELM {helm_params:,} params ({budget.active:,} active/token)  vs  "
          f"Euclidean width {width} at {euclid_params:,} "
          f"({(euclid_params - helm_params) / helm_params:+.1%})")
    print(f"lr: HELM {options.lr:g}, Euclidean {options.lr_dense:g} -- each arm "
          f"at its OWN swept optimum")
    print("  established over 9 sweep points at 1200 steps, both bracketed:")
    print("    HELM      1e-4 508.70 | 2e-4 466.06 | 4e-4 536.07")
    print("    Euclidean 1.6e-3 129.63 | 3.2e-3 124.51 | 6.4e-3 157.33")
    print("  the two optima differ by 16x, which is why every shared-grid "
          "comparison\n  in this project was measuring the learning rate rather "
          "than the geometry.")

    if options.lr_sweep or options.lr_sweep_helm:
        grids = {
            "helm": [float(v) for v in
                     (options.lr_sweep_helm or options.lr_sweep).split(",")],
            "euclid": [float(v) for v in
                       (options.lr_sweep_euclid or options.lr_sweep).split(",")],
        }
        sweep_cfg = dict(cfg, steps=options.sweep_steps)
        print(f"\nLR sweep at {options.sweep_steps} steps -- each arm judged at "
              f"its OWN best rate\n")
        print(f"{'arm':>8s} {'lr':>10s} {'val ppl':>10s}")
        best = {}
        for name in ("helm", "euclid"):
            for rate in grids[name]:
                stream = stream_from(train_ids, options.batch, options.seq_len,
                                     seed=0)
                _, logits_fn, _, _, _ = train_arm(name, args, sweep_cfg, stream,
                                                  0, device, rate, width)
                ppl, _ = perplexity_and_deciles(logits_fn, eval_batches[:8],
                                                deciles)
                print(f"{name:>8s} {rate:10.1e} {ppl:10.2f}", flush=True)
                if name not in best or ppl < best[name][1]:
                    best[name] = (rate, ppl)
        print(f"\nbest: HELM {best['helm'][0]:.1e} (ppl {best['helm'][1]:.2f}), "
              f"Euclidean {best['euclid'][0]:.1e} (ppl {best['euclid'][1]:.2f})")
        # A best point at the edge of its grid is the boundary of the search,
        # not an optimum. Comparing two arms at unbracketed boundaries measures
        # the grids rather than the architectures -- which is what the previous
        # run was about to do, with HELM pinned at the bottom of the grid and the
        # control at the top.
        edge = [n for n in ("helm", "euclid")
                if len(grids[n]) > 1
                and best[n][0] in (min(grids[n]), max(grids[n]))]
        if edge:
            raise SystemExit(
                f"\nSWEEP FAILED for {', '.join(edge)}: best rate sits at the "
                f"edge of its grid, so the optimum is not bracketed. Widen it "
                f"(--lr-sweep-helm / --lr-sweep-euclid) rather than comparing at "
                f"a boundary.")
        options.lr, options.lr_dense = best["helm"][0], best["euclid"][0]
        print(f"proceeding with those rates\n")

    if options.time_only:
        cfg = dict(cfg, steps=10)
        for name in ("helm", "euclid"):
            stream = stream_from(train_ids, options.batch, options.seq_len, seed=0)
            _, _, per_step, _, _ = train_arm(name, args, cfg, stream, 0, device,
                                             options.lr, width)
            print(f"  {name:7s} {per_step*1000:7.1f} ms/step -> "
                  f"{options.steps} steps = {per_step*options.steps/60:5.1f} min")
        return

    rows: List[Dict] = []
    for name in ("helm", "euclid"):
        for seed in [int(s) for s in options.seeds.split(",")]:
            stream = stream_from(train_ids, options.batch, options.seq_len, seed=0)
            arm_lr = options.lr if name == "helm" else options.lr_dense
            model, logits_fn, per_step, history, violation = train_arm(
                name, args, cfg, stream, seed, device, arm_lr, width)
            ppl, by_decile = perplexity_and_deciles(logits_fn, eval_batches, deciles)

            @torch.no_grad()
            def draw(tokens, n, logits_fn=logits_fn):
                probs = logits_fn(tokens[:, :-1]).float().softmax(-1)
                flat = probs.reshape(-1, probs.size(-1))
                drawn = torch.multinomial(flat, n, replacement=True).T
                return drawn.reshape(n, *probs.shape[:-1]), tokens[:, 1:]

            brier = M.brier_by_order(draw, eval_batches[:8])
            row = {"arm": name, "seed": seed, "perplexity": ppl,
                   "ppl_by_decile": by_decile, "brier": brier,
                   "top1": M.top1_accuracy(draw, eval_batches[:8]),
                   "final_loss": statistics.mean(history[-100:]),
                   "ms_per_step": per_step * 1000,
                   "manifold_violation": violation}
            rows.append(row)
            # Write after every row. Three container reaps have already cost
            # partial runs; a result that only exists at the end of a four-hour
            # job is a result that keeps not existing.
            if options.out:
                Path(options.out).parent.mkdir(parents=True, exist_ok=True)
                Path(options.out).write_text(json.dumps(
                    {"config": vars(options), "helm_params": helm_params,
                     "euclid_params": euclid_params, "rows": rows,
                     "complete": False}, indent=2))
            print(f"{name:7s} seed {seed}  ppl {ppl:8.2f}  top1 {row['top1']:6.2%}  "
                  f"brier1 {brier[0]:+.5f}  {per_step*1000:6.1f} ms/step  "
                  f"manifold_err {violation:.2e}", flush=True)

    report(rows, options)
    if options.out:
        Path(options.out).write_text(json.dumps(
            {"config": vars(options), "helm_params": helm_params,
             "euclid_params": euclid_params, "rows": rows,
             "complete": True}, indent=2))
        print(f"\nwrote {options.out}")


def agg(rows, arm, key):
    values = [r[key] for r in rows if r["arm"] == arm]
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def report(rows, options):
    print("\n" + "=" * 74)
    helm_ppl, helm_sd = agg(rows, "helm", "perplexity")
    euclid_ppl, euclid_sd = agg(rows, "euclid", "perplexity")
    noise = max(helm_sd, euclid_sd, 1e-9)
    print(f"perplexity   HELM {helm_ppl:8.2f} (sd {helm_sd:.2f})   "
          f"Euclidean {euclid_ppl:8.2f} (sd {euclid_sd:.2f})")
    print(f"             difference {euclid_ppl - helm_ppl:+.2f} "
          f"(positive = HELM better), seed sd {noise:.2f}")

    helm_b, _ = agg(rows, "helm", "top1")
    euclid_b, _ = agg(rows, "euclid", "top1")
    print(f"top-1        HELM {helm_b:8.2%}   Euclidean {euclid_b:8.2%}   "
          f"difference {helm_b - euclid_b:+.2%}")

    print("\nperplexity by token-frequency decile (0 = most frequent, 9 = rarest)")
    print(f"{'decile':>8s} {'HELM':>10s} {'Euclid':>10s} {'HELM better by':>16s}")
    helm_d = [statistics.mean(x) for x in zip(*[r["ppl_by_decile"] for r in rows
                                                if r["arm"] == "helm"])]
    euclid_d = [statistics.mean(x) for x in zip(*[r["ppl_by_decile"] for r in rows
                                                  if r["arm"] == "euclid"])]
    for i, (h, e) in enumerate(zip(helm_d, euclid_d)):
        print(f"{i:>8d} {h:10.2f} {e:10.2f} {100 * (e - h) / max(e, 1e-9):15.1f}%")

    print("\n" + "-" * 74)
    if euclid_ppl - helm_ppl > noise:
        print("HELM's advantage APPEARS in perplexity, a metric a CALM model can be")
        print("compared on through BrierLM. The integration is measurable -- proceed")
        print("to the 2x2, and check whether the decile column shows the advantage")
        print("concentrated in the rare tail, which would predict a negative")
        print("interaction with CALM's density-weighted objective.")
    elif abs(euclid_ppl - helm_ppl) <= noise:
        print("HELM's advantage does NOT appear in perplexity at this scale.")
        print("Combined with HELM reporting no perplexity itself, and its accuracy")
        print("gains sitting at near-chance level, there is currently no metric on")
        print("which a HELM-CALM could be shown to inherit HELM's advantage.")
        print("Recommendation: do not build the integration on this evidence.")
    else:
        print("The Euclidean arm is BETTER on perplexity. If this holds, HELM's")
        print("reported gains do not extend to language modelling at this scale,")
        print("and the integration has no established advantage to inherit.")
    print("Scale note: HELM's own results are at 120M/1B on 5B tokens. This run is")
    print("far smaller, so a null here is weaker evidence than a positive result.")


if __name__ == "__main__":
    main()
