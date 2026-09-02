#!/usr/bin/env python3
"""Run the 2x2, report the interaction, and refuse to over-claim.

See ``README.md`` for the design. In brief: four cells (hyperbolic or Euclidean
backbone) x (discrete or CALM head), matched on parameters, sharing data,
optimizer, schedule and seeds, so that

    interaction = (helm_calm - euclid_calm) - (helm_discrete - euclid_discrete)

is attributable to the combination rather than to either part. The runner
enforces two gates stated in advance:

1. **Floor gate** -- a cell scoring below the bigram lookup table has not
   learned the corpus and is reported as descriptive of nothing.
2. **Reproduction gate** -- if the geometry effect is absent in the *discrete*
   column, HELM has not been reproduced here and the CALM column cannot be
   interpreted. The runner says so instead of printing an interaction.

Tiers (``--tier``):

  0  CPU smoke, ~30 steps      does every cell run and every metric come out finite
  1  CPU study, hours          the full 2x2 with seeds; signs and a noise floor
  2  single GPU                the same design at the 120M preset
  3  multi-GPU                 HELM's published setting; the only comparable tier

Usage::

    python CALM/suite/run.py --tier 0
    python CALM/suite/run.py --tier 1 --seeds 0,1,2
    python CALM/suite/run.py --tier 2 --device cuda --out results_gpu.json
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
for _extra in (ROOT, ROOT / "CALM", ROOT / "CALM" / "experiments",
               ROOT / "CALM" / "suite"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import metrics as M  # noqa: E402
import probes as P  # noqa: E402
from helm_calm import PatchAutoencoder  # noqa: E402
from models import (CELLS, EuclideanCalm, EuclideanDiscrete,  # noqa: E402
                    build_helm_calm, build_helm_discrete, count_parameters,
                    describe, match_euclidean_width)
from corpus import batches_from, load, overlap_report, stream_from  # noqa: E402
from tests._config import tiny_args  # noqa: E402

TIERS = {
    0: dict(steps=30, ae_steps=50, seq_len=64, batch=4, dim=33, layers=2,
            heads=3, latent=16, train_batches=0, eval_batches=4),
    1: dict(steps=4000, ae_steps=3000, seq_len=128, batch=8, dim=65, layers=4,
            heads=5, latent=64, train_batches=0, eval_batches=16),
    2: dict(steps=20000, ae_steps=8000, seq_len=512, batch=16, dim=513,
            layers=12, heads=8, latent=128, train_batches=0, eval_batches=64),
    3: dict(steps=100000, ae_steps=20000, seq_len=2048, batch=32, dim=1025,
            layers=24, heads=16, latent=128, train_batches=0,
            eval_batches=256),
}


def helm_args(cfg, vocab: int):
    """A HELM config at the tier's shape, with the structural constraints kept.

    ``dim`` odd so ``dim - 1`` is even, rope dim odd, kv_lora_rank odd -- the
    relationships ``tests/_config.py`` documents. Getting these wrong produces a
    shape error, not a silently different model, which is the good case.
    """
    head = max(((cfg["dim"] - 1) // cfg["heads"]) // 2 * 2 + 1, 5)
    return tiny_args(vocab_size=vocab, dim=cfg["dim"], n_layers=cfg["layers"],
                     n_heads=cfg["heads"], max_seq_len=cfg["seq_len"],
                     original_seq_len=cfg["seq_len"], max_batch_size=cfg["batch"],
                     qk_nope_head_dim=head, qk_rope_head_dim=head,
                     v_head_dim=head, kv_lora_rank=(cfg["dim"] // 2) * 2 + 1,
                     inter_dim=2 * cfg["dim"], moe_inter_dim=cfg["dim"] + 31,
                     mice_inter_dim=cfg["dim"] + 31)


def train_autoencoder(train, cfg, patch, device, vocab, seed=0):
    torch.manual_seed(seed)
    ae = PatchAutoencoder(vocab, hidden=4 * cfg["dim"], latent_size=cfg["latent"],
                          patch_size=patch).to(device)
    optimizer = torch.optim.AdamW(ae.parameters(), lr=3e-3)
    generator = torch.Generator().manual_seed(seed)
    rows = train[:(train.numel() // patch) * patch].view(-1, patch)
    for _ in range(cfg["ae_steps"]):
        index = torch.randint(0, rows.size(0), (256,), generator=generator)
        loss, _ = ae.elbo(rows[index].to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return ae.freeze()


@torch.no_grad()
def autoencoder_ceiling(ae, data, patch, device):
    """CALM's stated precondition, and a hard cap on both CALM cells."""
    rows = data[:(data.numel() // patch) * patch].view(-1, patch)[:8192].to(device)
    mean, _ = ae.encode(rows)
    hit = ae.decode(mean).argmax(-1) == rows
    return hit.all(dim=-1).float().mean().item()


def run_cell(name, cfg, args, ae, train_stream, eval_batches, patch, device,
             seed, vocab, lr=1e-3):
    """Train one cell and collect everything the design asks of it."""
    torch.manual_seed(seed)
    is_helm = name.startswith("helm")
    is_calm = name.endswith("calm")

    if name == "helm_discrete":
        model = build_helm_discrete(args).to(device)
        step_loss = lambda t: _unwrap(model(t[:, :-1], labels=t[:, 1:]))
    elif name == "helm_calm":
        model = build_helm_calm(args, ae).to(device)
        step_loss = model.loss
    elif name == "euclid_discrete":
        model = EuclideanDiscrete(vocab, cfg["euclid_dim"], cfg["layers"],
                                  cfg["heads"], 2 * cfg["euclid_dim"],
                                  cfg["seq_len"]).to(device)
        step_loss = model.loss
    else:
        model = EuclideanCalm(vocab, cfg["euclid_dim"], cfg["layers"],
                              cfg["heads"], 2 * cfg["euclid_dim"], patch,
                              ae.latent_size, cfg["seq_len"]).to(device)
        step_loss = lambda t: model.loss(t, ae)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)
    model.train()
    history, grad_norms, nonfinite = [], [], 0
    started = time.time()
    for step in range(cfg["steps"]):
        loss = step_loss(next(train_stream).to(device))
        if not torch.isfinite(loss):
            nonfinite += 1
            optimizer.zero_grad()
            continue
        optimizer.zero_grad()
        loss.backward()
        grad_norms.append(
            torch.nn.utils.clip_grad_norm_(params, 1.0).item())
        optimizer.step()
        if hasattr(model, "retract_manifold_parameters"):
            model.retract_manifold_parameters()
        history.append(loss.item())
    seconds_per_step = (time.time() - started) / max(cfg["steps"], 1)
    model.eval()

    draw = _draw_fn(name, model, ae, patch)
    block_draw = _block_draw_fn(name, model, ae, patch)
    row = {
        "cell": name, "seed": seed, "geometry": "hyperbolic" if is_helm else "euclidean",
        "objective": "calm" if is_calm else "discrete",
        "top1_teacher_forced": M.top1_accuracy(draw, eval_batches),
        "brier": M.brier_by_order(draw, eval_batches),
        # The format-neutral comparison: next K tokens from complete blocks,
        # one AR step for CALM against K for the discrete column.
        "block": M.block_accuracy(block_draw, eval_batches[:4], patch,
                                  n_samples=8),
        "ar_steps_per_block": 1 if name.endswith("calm") else patch,
        "bpb": None,
        "final_loss": statistics.mean(history[-50:]) if history else float("nan"),
        "steps_to_threshold": M.steps_to_threshold(
            history, statistics.mean(history[:50]) * 0.5 if history else 0.0),
        "grad": M.gradient_percentiles(grad_norms),
        "nonfinite_steps": nonfinite,
        "seconds_per_step": seconds_per_step,
    }
    if not is_calm:
        logits = ((lambda t: _unwrap(model(t))) if is_helm else model.logits)
        row["bpb"] = M.bits_per_byte(logits, eval_batches)
    row.update(_probe(name, model, ae, eval_batches[0], patch, is_helm, is_calm))
    return row


def _unwrap(out):
    return out[0] if isinstance(out, tuple) else out


def _block_draw_fn(name, model, ae, patch):
    """Next-K-tokens from complete blocks only -- no ground truth inside the block.

    The asymmetry this removes: a teacher-forced discrete model is shown the
    ground-truth tokens CALM never sees. Here both columns get prior complete blocks and
    nothing else, so the comparison is in the format CALM targets rather than
    the one it exists to escape.
    """
    if name.endswith("calm"):
        # One latent, decoded to K tokens: a single autoregressive step.
        base = _draw_fn(name, model, ae, patch)

        def draw(tokens, n):
            samples, targets = base(tokens, n)
            # (n, rows, S') -> (n, blocks, K)
            return (samples.reshape(n, -1, patch), targets.reshape(-1, patch))
        return draw

    logits_fn = ((lambda t: _unwrap(model(t))) if name.startswith("helm")
                 else model.logits)

    @torch.no_grad()
    def draw(tokens, n):
        """Free-run K steps, feeding back the model's own samples."""
        n_blocks = tokens.size(1) // patch
        prefix_len = (n_blocks - 1) * patch
        context = tokens[:, :prefix_len]
        targets = tokens[:, patch:n_blocks * patch].reshape(-1, patch)
        drawn = []
        for _ in range(n):
            running = context
            emitted = []
            for _ in range(patch):
                probs = logits_fn(running).float().softmax(-1)[:, -1]
                nxt = torch.multinomial(probs, 1)
                emitted.append(nxt)
                running = torch.cat([running, nxt], dim=1)
            drawn.append(torch.cat(emitted, dim=1))
        # (n, batch, K); the block predicted is the one after the prefix
        samples = torch.stack(drawn)
        return samples.reshape(n, -1, patch), targets[-samples.shape[1]:]
    return draw


def _draw_fn(name, model, ae, patch):
    if name == "helm_calm":
        def draw(tokens, n):
            samples, targets = model.sample_tokens(tokens, n_samples=n)
            rows = tokens.size(0)
            return samples.reshape(n, rows, -1), targets.reshape(rows, -1)
        return draw
    if name == "euclid_calm":
        return lambda tokens, n: model.draw(tokens, ae, n)

    logits_fn = ((lambda t: _unwrap(model(t))) if name.startswith("helm")
                 else model.logits)

    @torch.no_grad()
    def draw(tokens, n):
        probs = logits_fn(tokens[:, :-1]).float().softmax(-1)
        flat = probs.reshape(-1, probs.size(-1))
        drawn = torch.multinomial(flat, n, replacement=True).T
        return drawn.reshape(n, *probs.shape[:-1]), tokens[:, 1:]
    return draw


@torch.no_grad()
def _probe(name, model, ae, tokens, patch, is_helm, is_calm):
    """The why-columns. Token-level and patch-level structure, and numerics."""
    out = {}
    manifold = getattr(getattr(model, "backbone", model), "manifold_hidden", None)
    if is_calm and is_helm:
        hidden = model.hidden_states(model._aligned(tokens)[0])
        token_points = model.backbone.embed(tokens)
    elif is_calm:
        hidden = model.hidden_states(model.aligned(tokens)[0])
        token_points = model.embed(tokens)
    elif is_helm:
        hidden = _unwrap(model(tokens))
        token_points = model.embed(tokens)
    else:
        hidden = model.backbone(tokens)
        token_points = model.backbone.embed(tokens)

    out["numerics"] = P.numerics(hidden)
    out["effective_rank"] = M.effective_rank(hidden)
    if is_helm:
        from helm.hypercore.manifolds import Lorentz
        man = Lorentz(1.0)
        out["radius"] = P.radius_profile(token_points, man)
        if is_calm:
            out["hierarchy"] = P.hierarchy_flattening(
                token_points, model.patch_embed(token_points), man)
    else:
        out["radius"] = P.radius_profile(token_points)
        if is_calm:
            b, s, _ = token_points.shape
            patched = model.patch_embed(
                token_points.reshape(b, s // patch, -1))
            out["hierarchy"] = P.hierarchy_flattening(token_points, patched)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", type=int, default=0, choices=sorted(TIERS))
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--patch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--corpus", default="wikitext2",
                        choices=("wikitext2", "ptb"))
    parser.add_argument("--level", default="byte", choices=("byte", "word"))
    parser.add_argument("--steps", type=int, default=0,
                        help="override the tier's step count")
    parser.add_argument("--limit", type=int, default=0,
                        help="truncate each split; smoke tests only, and it "
                             "changes what is being measured")
    parser.add_argument("--cells", default=",".join(CELLS))
    parser.add_argument("--out", default="")
    options = parser.parse_args()

    cfg = dict(TIERS[options.tier])
    if options.steps:
        cfg["steps"] = options.steps
    device = torch.device(options.device)
    seeds = [int(v) for v in options.seeds.split(",")]
    cells = options.cells.split(",")

    corpus = load(options.corpus, options.level, options.limit or None)
    vocab = corpus.vocab_size
    train, valid = corpus.train, corpus.valid
    eval_batches = [b.to(device) for b in batches_from(
        valid, cfg["batch"], cfg["seq_len"], cfg["eval_batches"], seed=1)]

    args = helm_args(cfg, vocab)
    def make_stream():
        """A fresh, identically-seeded stream per cell over the whole split."""
        return stream_from(train, cfg["batch"], cfg["seq_len"], seed=0)

    ae = train_autoencoder(train, cfg, options.patch, device, vocab)
    ceiling = autoencoder_ceiling(ae, valid, options.patch, device)

    # Match the Euclidean width to HELM's parameter count before anything trains.
    width, euclid_params, helm_params = match_euclidean_width(
        lambda: build_helm_discrete(args),
        lambda w: EuclideanDiscrete(vocab, w, cfg["layers"], cfg["heads"],
                                    2 * w, cfg["seq_len"]))
    cfg["euclid_dim"] = width
    budget = describe(build_helm_discrete(args), args, cfg["seq_len"], is_helm=True)

    base = M.lookup_baselines(train, [b.cpu() for b in eval_batches], vocab)
    seen = cfg["steps"] * cfg["batch"] * cfg["seq_len"]
    print(f"  training stream: {seen:,} tokens over {cfg['steps']:,} steps "
          f"= {seen / train.numel():.2f} epochs of the split")
    print(f"tier {options.tier}  K={options.patch}  seeds {seeds}  device {device}")
    print(corpus.describe())
    print(f"  digests {corpus.digests}")
    if not options.limit:
        overlap = overlap_report(corpus)["verbatim_fraction"]
        print(f"  16-gram verbatim overlap valid->train: {overlap:.2%}"
              f"{'  -- some held-out accuracy is recall' if overlap > 0.05 else ''}")
    print(f"HELM {helm_params:,} params ({budget.active:,} active/token) vs "
          f"Euclidean width {width} at {euclid_params:,} "
          f"({(euclid_params - helm_params) / helm_params:+.1%})")
    print(f"autoencoder ceiling (held out, per patch): {ceiling:.2%}  "
          f"-- a hard cap on both CALM cells")
    print(f"floor: bigram top-1 {base['bigram_top1']:.2%}, BPB "
          f"{base['bigram_bpb']:.4f}; uniform BPB {base['uniform_bpb']:.2f}\n")

    rows: List[Dict] = []
    for cell in cells:
        for seed in seeds:
            row = run_cell(cell, cfg, args, ae, make_stream(), eval_batches,
                           options.patch, device, seed, vocab, options.lr)
            rows.append(row)
            bpb = f"{row['bpb']:.4f}" if row["bpb"] is not None else "n/a"
            print(f"{cell:16s} seed {seed}  tf-top1 {row['top1_teacher_forced']:6.2%}  "
                  f"block {row['block']['exact_block']:6.2%}  "
                  f"tok-in-blk {row['block']['token_in_block']:6.2%}  "
                  f"AR/blk {row['ar_steps_per_block']}  BPB {bpb:>8s}  "
                  f"brier1 {row['brier'][0]:+.5f}  rank {row['effective_rank']:6.1f}  "
                  f"NaN {row['nonfinite_steps']:3d}", flush=True)

    report(rows, base, ceiling, options)
    if options.out:
        Path(options.out).write_text(json.dumps(
            {"config": {**cfg, "tier": options.tier, "patch": options.patch,
                        "seeds": seeds},
             "baselines": base, "ae_ceiling": ceiling, "rows": rows}, indent=2))
        print(f"\nwrote {options.out}")


def mean_of(rows, cell, key="top1_teacher_forced"):
    values = [r[key] for r in rows if r["cell"] == cell]
    return statistics.mean(values) if values else float("nan")


def spread_of(rows, cell, key="top1_teacher_forced"):
    values = [r[key] for r in rows if r["cell"] == cell]
    return statistics.stdev(values) if len(values) > 1 else 0.0


def report(rows, base, ceiling, options):
    """Apply the gates from README section 6, in order, and stop at the first failure."""
    print("\n" + "=" * 72)
    have = {r["cell"] for r in rows}

    below = [c for c in have if mean_of(rows, c) < base["bigram_top1"]]
    invalid = [c for c in have if any(r["nonfinite_steps"] for r in rows
                                      if r["cell"] == c)]
    if invalid:
        print(f"INVALID: {', '.join(sorted(invalid))} produced non-finite steps. "
              f"Those rows describe nothing; fix before reading anything else.")
    if below:
        print(f"BELOW FLOOR: {', '.join(sorted(below))} scored under the bigram "
              f"lookup table ({base['bigram_top1']:.2%}). A model that loses to "
              f"counting has not learned the corpus, and its row is not a "
              f"comparison.")

    # The floor gate has to BLOCK, not warn. The tier-0 smoke run had all four
    # cells below the bigram table and still printed "INTERACTION: +3.89% --
    # hyperbolic geometry helps MORE under CALM's objective", which is exactly
    # the sentence someone would quote. A difference between four models that
    # have each failed to learn the corpus is a difference between four kinds of
    # noise.
    if set(below) >= (have & {"helm_discrete", "euclid_discrete"}) and below:
        print("\nNo interaction reported: the discrete column is below the floor, "
              "so its geometry effect is a difference between two models that "
              "have not learned the corpus. Train longer or at a larger tier.")
        return
    if invalid:
        print("\nNo interaction reported: at least one cell is numerically "
              "invalid.")
        return

    # A single seed gives no noise estimate, so every gate that compares an
    # effect against seed spread would pass trivially. Refuse rather than
    # compare an effect against a spread of zero.
    seeds_per_cell = min(len({r["seed"] for r in rows if r["cell"] == c})
                         for c in have)
    if seeds_per_cell < 2:
        print(f"\nNo interaction reported: {seeds_per_cell} seed per cell gives "
              f"no noise estimate, and an effect cannot be judged against a "
              f"spread of zero. Re-run with --seeds 0,1,2.")
        return

    if not {"helm_discrete", "euclid_discrete"} <= have:
        print("\nThe discrete column is incomplete, so the reproduction gate "
              "cannot be applied and no interaction is reported.")
        return

    geometry_discrete = mean_of(rows, "helm_discrete") - mean_of(rows, "euclid_discrete")
    noise = max(spread_of(rows, "helm_discrete"),
                spread_of(rows, "euclid_discrete"))
    print(f"\ngeometry effect, discrete column: {geometry_discrete:+.2%} "
          f"(seed sd {noise:.2%})")

    if abs(geometry_discrete) < noise:
        print("REPRODUCTION GATE FAILED: the hyperbolic advantage HELM reports is "
              "not present here at this scale, so the CALM column cannot be "
              "attributed to geometry. No interaction reported -- it would not "
              "mean what it appears to mean.")
        return

    if not {"helm_calm", "euclid_calm"} <= have:
        print("CALM column incomplete; no interaction reported.")
        return

    geometry_calm = mean_of(rows, "helm_calm") - mean_of(rows, "euclid_calm")
    interaction = geometry_calm - geometry_discrete
    print(f"geometry effect, CALM column:     {geometry_calm:+.2%}")
    print(f"INTERACTION:                      {interaction:+.2%}")
    if abs(interaction) < noise:
        verdict = ("~0: patching neither helps nor harms the hyperbolic "
                   "advantage at this scale")
    elif interaction < 0:
        verdict = ("< 0: patching costs the hyperbolic advantage -- the concrete "
                   "form of the HIERARCHY.md worry. Check flattening_ratio.")
    else:
        verdict = ("> 0: hyperbolic geometry helps MORE under CALM's objective. "
                   "The strongest case for HELM-CALM, and the one to doubt "
                   "hardest -- check radius, pinned and nonfinite first.")
    print(f"  {verdict}")

    ratios = [r["hierarchy"]["flattening_ratio"] for r in rows
              if "hierarchy" in r and r["cell"] == "helm_calm"]
    if ratios:
        print(f"\nflattening ratio (patch delta / token delta), helm_calm: "
              f"{statistics.mean(ratios):.3f}  -- above 1 means patching "
              f"flattened the hierarchy")
    print(f"\nCeiling reminder: no CALM cell can exceed the autoencoder's "
          f"{ceiling:.2%} per-patch reconstruction, whatever its backbone.")
    print("Scale reminder: tier 0 and 1 are small-scale studies. Only tier 3 "
          "produces numbers comparable to HELM's published table.")


if __name__ == "__main__":
    main()
