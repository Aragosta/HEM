#!/usr/bin/env python3
"""T8, the part that fits: how much does top-k-over-the-sequence routing leak?

Mixture-of-Depths selects the top `k` tokens *of the sequence* to receive a
layer's full computation. That is a comparison BETWEEN tokens, so a token's
selection can depend on tokens that come after it -- which in an autoregressive
model is a leak: at generation time those tokens do not exist, so the routing
decision cannot be reproduced, and any quality measured with it is inflated.

The published fix is to train a causal predictor of the routing decision and use
it at inference. Whether that matters depends on how different the two decisions
are, and that is measurable with **no training at all**: run one forward pass and
compare, for each position `t` and each layer,

* the selection computed from the whole sequence (what MoD does at train time),
* the selection computed from the prefix `1..t` only (what is available at
  generation time).

No model quality is involved, so an untrained model is the right instrument: the
question is about the selection rule's dependence structure, not about what a
good router would choose. Reported at several capacities, since a leak that
matters at `c=0.125` may not at `c=0.75`.

**Prediction, registered before the run:** disagreement is large at small
capacity and falls as capacity rises, because at `c` near 1 almost every token is
selected under either rule and there is nothing to disagree about. If
disagreement at `c=0.5` is below a few percent the leak is a technicality; if it
is tens of percent, any MoD number reported without the causal-predictor
evaluation is not a language-modelling number.

Usage::

    python CALM/suite/mod_audit.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
for _extra in (ROOT, ROOT / "CALM", ROOT / "CALM" / "suite"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from bpe import encode_split, train_or_load  # noqa: E402
from corpus import batches_from  # noqa: E402
from hybrid import HybridDecoder  # noqa: E402


@torch.no_grad()
def disagreement(scores: torch.Tensor, capacity: float) -> float:
    """Fraction of positions whose selection changes when the future is hidden.

    `scores` is (batch, seq). The full-sequence rule takes the top `c*n` over the
    whole row; the causal rule asks, at each `t`, whether `t` is in the top
    `c*(t+1)` of the prefix -- the best a generation-time router could do.
    """
    batch, n = scores.shape
    keep = max(1, int(round(capacity * n)))
    full = torch.zeros_like(scores, dtype=torch.bool)
    full.scatter_(1, scores.topk(keep, dim=1).indices, True)

    causal = torch.zeros_like(full)
    for t in range(n):
        prefix = scores[:, :t + 1]
        take = max(1, int(round(capacity * (t + 1))))
        threshold = prefix.topk(take, dim=1).values[:, -1]
        causal[:, t] = scores[:, t] >= threshold
    return (full != causal).float().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--kv-latent", type=int, default=48)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--vocab", type=int, default=16000)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="CALM/suite/mod_audit_results.json")
    o = parser.parse_args()

    tokenizer = train_or_load("wikitext2", o.vocab)
    valid = encode_split(tokenizer, "wikitext2", "valid")
    vocab = tokenizer.get_vocab_size()
    batches = batches_from(valid, o.batch, o.seq_len, o.batches, seed=1)

    torch.manual_seed(o.seed)
    model = HybridDecoder(vocab, dim=o.dim, layers=o.layers,
                          heads=max(o.dim // o.head_dim, 1), head_dim=o.head_dim,
                          kv_latent=o.kv_latent, ffn="dense",
                          max_seq_len=o.seq_len).eval()
    # A depth router is a linear map from the residual stream to one score per
    # token. Untrained is correct here: the question is the selection rule's
    # dependence on the future, which no amount of training changes.
    routers = [torch.nn.Linear(o.dim, 1, bias=False) for _ in range(o.layers)]

    print("T8 (audit)  does top-k-over-the-sequence routing depend on the future?")
    print(f"    seq_len {o.seq_len}, {o.layers} layers, "
          f"{o.batches} batches x {o.batch} sequences, untrained routers\n")
    print(f"    {'capacity':>9s} {'disagreement':>13s} {'per-layer':>34s}")

    rows = []
    with torch.no_grad():
        streams = []
        for tokens in batches:
            x = model.embed(tokens[:, :-1])
            layer_scores = []
            for block in model.blocks:
                layer_scores.append(x)
                x = block(x)
            streams.append(layer_scores)

        for capacity in (0.125, 0.25, 0.5, 0.75):
            per_layer = []
            for layer in range(o.layers):
                scores = torch.cat([routers[layer](s[layer]).squeeze(-1)
                                    for s in streams], dim=0)
                per_layer.append(disagreement(scores, capacity))
            row = {"capacity": capacity, "mean": statistics.mean(per_layer),
                   "per_layer": per_layer}
            rows.append(row)
            print(f"    {capacity:9.3f} {row['mean']:12.2%}  "
                  f"{[f'{v:.1%}' for v in per_layer]}")

    at_half = next(r["mean"] for r in rows if r["capacity"] == 0.5)
    print()
    verdict = ("a technicality at this scale" if at_half < 0.03 else
               "REAL: any MoD number reported without a causal-predictor "
               "evaluation is inflated")
    print(f"    disagreement at c=0.5 is {at_half:.2%} -- {verdict}")
    print("    Selection rule only. No model quality is measured or implied.")

    if o.out:
        Path(o.out).write_text(json.dumps({"config": vars(o), "rows": rows},
                                          indent=1))
        print(f"    wrote {o.out}")


if __name__ == "__main__":
    main()
