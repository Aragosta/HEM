"""Is a failure to learn the task the task's fault or the model's?

A 30-line textbook pre-norm transformer, trained on exactly the same batches
through exactly the same loop as `probe_learnable.py`. If this learns the task
and `model.py` does not, the fault is in `model.py`; if neither learns it, the
fault is in the task or the budget. Every ablation of the real model can then
be read against a control that has no AttnRes, no MoE, no loop and no
re-injection.

    python probe_reference.py --hops 1 --steps 400 --lr 3e-3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import Config, Recurrent, apply_rotary, rotary_table       # noqa: E402
from tasks import HopSpec, hop_batch, hop_eval_set                    # noqa: E402


class Reference(nn.Module):
    def __init__(self, vocab, dim, layers, heads, seq_len):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(nn.ModuleDict({
                "n1": nn.RMSNorm(dim), "qkv": nn.Linear(dim, 3 * dim, bias=False),
                "o": nn.Linear(dim, dim, bias=False), "n2": nn.RMSNorm(dim),
                "g": nn.Linear(dim, 3 * dim, bias=False),
                "u": nn.Linear(dim, 3 * dim, bias=False),
                "d": nn.Linear(3 * dim, dim, bias=False)}))
        self.norm = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.heads = heads
        self.register_buffer("rot", rotary_table(dim // heads, seq_len + 8))

    def forward(self, tokens):
        x = self.embed(tokens)
        b, n, d = x.shape
        for blk in self.blocks:
            h = blk["n1"](x)
            q, k, v = blk["qkv"](h).chunk(3, -1)
            shape = (b, n, self.heads, d // self.heads)
            q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
            q, k = apply_rotary(q, self.rot[:n]), apply_rotary(k, self.rot[:n])
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + blk["o"](a.transpose(1, 2).reshape(b, n, d))
            h = blk["n2"](x)
            x = x + blk["d"](F.silu(blk["g"](h)) * blk["u"](h))
        return self.head(self.norm(x))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hops", default="1")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--entities", type=int, default=12)
    ap.add_argument("--queries", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--sorted", action="store_true")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    hops = tuple(int(h) for h in args.hops.split(","))
    spec = HopSpec(n_entities=args.entities, hops=hops, queries=args.queries,
                   sorted_pairs=args.sorted)
    torch.manual_seed(0)
    model = Reference(spec.vocab_size, args.dim, args.layers, 4, spec.seq_len)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    g = torch.Generator().manual_seed(1234)
    positions = spec.answer_positions()
    eval_set = hop_eval_set(spec, 128, 99)
    print(f"reference transformer, {args.layers} blocks, "
          f"{sum(p.numel() for p in model.parameters()):,} params | "
          f"{spec.describe()}", flush=True)

    for step in range(args.steps):
        tokens, target, _ = hop_batch(spec, args.batch, g)
        logits = model(tokens)[:, positions]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               target.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % max(1, args.steps // 10) == 0:
            with torch.no_grad():
                accs = {}
                for h, (tok, tgt, _) in eval_set.items():
                    pred = model(tok)[:, positions]
                    accs[h] = (pred.argmax(-1) == tgt).float().mean().item()
            print(f"  step {step + 1:5d} loss {loss.item():.3f} "
                  + " ".join(f"h{h}={a:.3f}" for h, a in accs.items()), flush=True)


if __name__ == "__main__":
    main()
