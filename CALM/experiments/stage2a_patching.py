#!/usr/bin/env python3
"""Stage 2a: does the K>1 patching path work end to end?

Stage 1 established that a hyperbolic backbone trains under CALM's energy score,
but only at K=1 -- deliberately, so compression could not confound trainability.
Stage 2 changes three things at once: the input is patched (K token embeddings
collapse to one vector), the sequence shortens by K, and the target becomes a
patch latent rather than a token latent. This runs that plumbing on the toy task
before any of it costs GPU time.

The HELM-specific part is the patch embedding. CALM concatenates K Euclidean
token embeddings and projects them::

    inputs_embeds = embed_tokens(ids).reshape(B, seq // K, -1)
    inputs_embeds = embed_proj(inputs_embeds)

HELM's embeddings are Lorentz vectors, and concatenating them does not give a
point on any hyperboloid. Here the K *space-like* parts are concatenated -- which
is a point in a higher-dimensional Minkowski space once its time coordinate is
recomputed -- and a LorentzLinear maps that back down onto the model's manifold.
That keeps every activation on a manifold, which is the property HELM exists to
maintain.

Note on the comparison: a token-level baseline sees every preceding *token*,
while the patched model only sees complete preceding *patches*. At K=4 it is
predicting up to four tokens ahead with no intermediate context. That handicap is
intrinsic to the method -- it is what buys 4x fewer autoregressive steps -- and
is not a defect of this implementation.

Usage: python CALM/experiments/stage2a_patching.py --patches 1 2 4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helm.hypercore.manifolds import Lorentz  # noqa: E402
from helm.hypercore.nn.linear.lorentz_linear import LorentzLinear  # noqa: E402
from helm.modules.helm_mice import HelmMiCE  # noqa: E402
from stage0_autoencoder import TinyAutoencoder  # noqa: E402
from stage1_energy_head import CalmHead, energy_score, make_batches  # noqa: E402
from tests._config import tiny_args  # noqa: E402


class LorentzPatchEmbedding(nn.Module):
    """Collapse K Lorentz token vectors into one, staying on the manifold."""

    def __init__(self, manifold, dim: int, patch: int):
        super().__init__()
        self.manifold = manifold
        self.patch = patch
        self.dim = dim
        # Input is a Lorentz vector of width patch*(dim-1)+1; output is width dim.
        self.proj = LorentzLinear(manifold, patch * (dim - 1) + 1, dim - 1)

    def forward(self, tokens_on_manifold: torch.Tensor) -> torch.Tensor:
        """(B, S, dim) -> (B, S // patch, dim)."""
        space = tokens_on_manifold[..., 1:]
        batch, seqlen, width = space.shape
        space = space.reshape(batch, seqlen // self.patch, self.patch * width)
        time = (space.square().sum(-1, keepdim=True) + self.manifold.c).clamp_min(1e-8).sqrt()
        return self.proj(torch.cat([time, space], dim=-1))


def patched_hidden(model, patch_embed, tokens):
    """Run HELM's backbone over patched inputs; return per-patch hidden states."""
    embedded = model.embed(tokens)                      # (B, S, dim), on-manifold
    h = patch_embed(embedded)                           # (B, S/K, dim)
    freqs_cis = model.freqs_cis[:h.size(1)]
    for layer in model.layers:
        out = layer(h, 0, freqs_cis, None, True, None)
        h = out[0] if isinstance(out, tuple) else out
    return model.norm(model.final_proj(h, return_space=True), space_only=True)


def pretrain_patch_autoencoder(vocab, batches, patch, steps=800, hidden=128,
                               latent=32, kl_weight=1e-3, seed=0):
    """Train the frozen K-token autoencoder; report reconstruction accuracy."""
    torch.manual_seed(seed)
    model = TinyAutoencoder(vocab, hidden=hidden, latent=latent, patch=patch, layers=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    flat = batches.reshape(-1)
    usable = (flat.numel() // patch) * patch
    windows = flat[:usable].view(-1, patch)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        rows = torch.randint(0, windows.size(0), (128,), generator=generator)
        loss, _, _ = model(windows[rows], kl_weight=kl_weight)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        mean, _ = model.encode(windows)
        accuracy = (model.decode(mean).argmax(-1) == windows).float().mean().item()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, accuracy


def run(args, batches, autoencoder, patch, steps, lr, seed, num_samples=8,
        eval_samples=32):
    """Train CALM-on-HELM at the given patch size; return next-token accuracy."""
    torch.manual_seed(seed)
    manifold = Lorentz(1.0)
    model = HelmMiCE(args, manifold, manifold, manifold)
    del model.head
    patch_embed = LorentzPatchEmbedding(manifold, args.dim, patch)
    head = CalmHead(args.dim, autoencoder.latent_size)

    embedding = model.embed.embedding
    others = [p for n, p in model.named_parameters()
              if p.requires_grad and not n.endswith("embed.embedding")]
    trainable = others + list(patch_embed.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    embed_opt = torch.optim.AdamW([embedding], lr=lr)

    model.train()
    head.train()
    losses, start = [], time.perf_counter()
    for step in range(steps):
        tokens = batches[step % len(batches)]
        n_patches = tokens.size(1) // patch
        # Patch p predicts patch p+1, so inputs are patches 0..P-2.
        inputs = tokens[:, :(n_patches - 1) * patch]
        targets = tokens[:, patch:n_patches * patch].reshape(-1, patch)

        with torch.no_grad():
            mean, log_std = autoencoder.encode(targets)

        hidden = patched_hidden(model, patch_embed, inputs).reshape(-1, args.dim)
        samples = head.sample(hidden.unsqueeze(0).expand(num_samples, -1, -1))
        loss = -energy_score(samples, mean, log_std).mean()

        optimizer.zero_grad()
        embed_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable + [embedding], 1.0)
        optimizer.step()
        embed_opt.step()
        with torch.no_grad():                          # keep the embedding on the manifold
            space = embedding[..., 1:]
            embedding.copy_(torch.cat(
                [(space.square().sum(-1, keepdim=True) + 1.0).sqrt(), space], dim=-1))
        losses.append(loss.item())

    model.eval()
    head.eval()
    correct = total = 0
    with torch.no_grad():
        for tokens in batches:
            n_patches = tokens.size(1) // patch
            inputs = tokens[:, :(n_patches - 1) * patch]
            targets = tokens[:, patch:n_patches * patch].reshape(-1, patch)
            hidden = patched_hidden(model, patch_embed, inputs).reshape(-1, args.dim)
            decoded = autoencoder.decode(
                head.sample(hidden.unsqueeze(0).expand(eval_samples, -1, -1))).argmax(-1)
            voted = torch.mode(decoded, dim=0).values
            correct += (voted == targets).sum().item()
            total += targets.numel()
    return losses, correct / total, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patches", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    cli = parser.parse_args()

    args = tiny_args()
    batches = make_batches(args.vocab_size, n_batches=16)

    print("=" * 78)
    print("STAGE 2a -- the K>1 patching path, end to end")
    print("=" * 78)
    print(f"\nbackbone: HELM-MiCE dim={args.dim}, {args.n_layers} layers, "
          f"vocab={args.vocab_size}")
    print(f"{cli.steps} steps, lr={cli.lr}, chance = {1 / args.vocab_size:.2%}\n")

    print(f"{'K':>3s} {'AE recon':>9s} {'positions':>10s} {'seeds':>22s} "
          f"{'mean acc':>9s} {'loss':>16s}")
    print("-" * 78)
    for patch in cli.patches:
        autoencoder, ae_accuracy = pretrain_patch_autoencoder(
            args.vocab_size, batches, patch)
        accs, first_last = [], None
        for seed in cli.seeds:
            losses, accuracy, _ = run(args, batches, autoencoder, patch,
                                      cli.steps, cli.lr, seed)
            accs.append(accuracy)
            if first_last is None:
                first_last = (losses[0], losses[-1])
        positions = batches.shape[-1] // patch
        cells = " ".join(f"{a:6.2%}" for a in accs)
        print(f"{patch:3d} {ae_accuracy:8.2%} {positions:10d} {cells:>22s} "
              f"{sum(accs) / len(accs):8.2%} {first_last[0]:7.2f} ->{first_last[1]:6.2f}")

    print("\nA token-level model sees every preceding token; at K=4 this one sees")
    print("only complete preceding patches, so it predicts up to 4 tokens ahead")
    print("blind. That handicap is what buys 4x fewer autoregressive steps.")


if __name__ == "__main__":
    main()
