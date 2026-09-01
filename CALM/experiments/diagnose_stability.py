#!/usr/bin/env python3
"""Why is CALM-on-HELM unstable across seeds?

Stage 1 found that a hyperbolic backbone trains under CALM's energy score, but
with a 27-point spread across three seeds where a Euclidean control spans 1.2.
This script tests the candidate explanations rather than guessing between them.

    A. evaluation noise -- 8-sample majority voting may just be a noisy estimator
       of the mode, in which case the spread is in the *metric*, not the model.
    B. manifold drift -- HELM's token embedding is a ManifoldParameter that lives
       on the hyperboloid and is meant to be optimized with a Riemannian
       optimizer. Stage 1 used plain AdamW on everything, which walks it off the
       manifold.
    C. gradient scale -- the energy loss and cross-entropy differ by an order of
       magnitude, so a gradient clip tuned for one may be doing something very
       different to the other.
    D. learning rate -- inherited from the discrete model and never tuned.

Usage: python CALM/experiments/diagnose_stability.py [--steps 5000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helm.hypercore.manifolds import Lorentz  # noqa: E402
from helm.modules.helm_mice import HelmMiCE  # noqa: E402
from stage1_energy_head import (CalmHead, backbone_hidden, energy_score,  # noqa: E402
                                make_batches, pretrain_autoencoder)
from tests._config import tiny_args  # noqa: E402


def manifold_violation(model, manifold_c=1.0):
    """How far the token embedding has drifted off the hyperboloid.

    Points on the manifold satisfy <x, x>_L = -c exactly. A ManifoldParameter
    updated by a Euclidean optimizer does not stay there.
    """
    embedding = model.embed.embedding
    squared = embedding.detach() ** 2
    quad = -squared[..., :1] + squared[..., 1:].sum(-1, keepdim=True)
    return (quad + manifold_c).abs()


def train_calm(args, batches, autoencoder, steps, lr, seed, num_samples=8,
               clip=1.0, riemannian=False, track=False):
    """Train CALM-on-HELM, optionally keeping the embedding on the manifold."""
    torch.manual_seed(seed)
    model = HelmMiCE(args, Lorentz(1.0), Lorentz(1.0), Lorentz(1.0))
    del model.head
    head = CalmHead(args.dim, autoencoder.latent_size)

    embedding = model.embed.embedding
    others = [p for n, p in model.named_parameters()
              if p.requires_grad and not n.endswith("embed.embedding")]
    optimizer = torch.optim.AdamW(others + list(head.parameters()), lr=lr)
    embed_opt = torch.optim.AdamW([embedding], lr=lr)

    model.train()
    head.train()
    grad_norms = []
    for step in range(steps):
        tokens = batches[step % len(batches)]
        targets = tokens[:, 1:].reshape(-1)
        with torch.no_grad():
            mean, log_std = autoencoder.encode(targets)
        hidden = backbone_hidden(model, tokens)[:, :-1].reshape(-1, args.dim)
        samples = head.sample(hidden.unsqueeze(0).expand(num_samples, -1, -1))
        loss = -energy_score(samples, mean, log_std).mean()

        optimizer.zero_grad()
        embed_opt.zero_grad()
        loss.backward()
        params = others + list(head.parameters()) + [embedding]
        norm = torch.nn.utils.clip_grad_norm_(params, clip)
        if track:
            grad_norms.append(float(norm))
        optimizer.step()
        embed_opt.step()
        if riemannian:
            # Retraction: project the embedding back onto the hyperboloid, which
            # is what a Riemannian optimizer would maintain.
            with torch.no_grad():
                space = embedding[..., 1:]
                time = (space.square().sum(-1, keepdim=True) + 1.0).sqrt()
                embedding.copy_(torch.cat([time, space], dim=-1))
    return model, head, grad_norms


@torch.no_grad()
def accuracy(model, head, autoencoder, batches, args, eval_samples):
    """Next-token accuracy by majority vote over a pool of `eval_samples`."""
    model.eval()
    head.eval()
    correct = total = 0
    for tokens in batches:
        targets = tokens[:, 1:].reshape(-1)
        hidden = backbone_hidden(model, tokens)[:, :-1].reshape(-1, args.dim)
        decoded = autoencoder.decode(
            head.sample(hidden.unsqueeze(0).expand(eval_samples, -1, -1))).argmax(-1)
        voted = torch.mode(decoded, dim=0).values
        correct += (voted == targets).sum().item()
        total += targets.numel()
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    cli = parser.parse_args()

    args = tiny_args()
    batches = make_batches(args.vocab_size, n_batches=16)
    autoencoder, ae_accuracy = pretrain_autoencoder(args.vocab_size, batches)
    print(f"frozen K=1 autoencoder: reconstruction {ae_accuracy:.2%}\n")

    print("=" * 74)
    print("A + B: is the spread in the metric, or in the model?")
    print("=" * 74)
    print(f"{'seed':>5s} {'riemannian':>11s} {'acc@8':>8s} {'acc@32':>8s} {'acc@128':>9s} "
          f"{'manifold err':>13s} {'grad norm':>10s}")
    for riemannian in (False, True):
        for seed in cli.seeds:
            model, head, norms = train_calm(args, batches, autoencoder, cli.steps,
                                            3e-3, seed, riemannian=riemannian, track=True)
            accs = [accuracy(model, head, autoencoder, batches, args, n)
                    for n in (8, 32, 128)]
            violation = manifold_violation(model).max().item()
            median_norm = sorted(norms)[len(norms) // 2]
            print(f"{seed:5d} {str(riemannian):>11s} {accs[0]:7.2%} {accs[1]:7.2%} "
                  f"{accs[2]:8.2%} {violation:13.3e} {median_norm:10.2f}")

    print()
    print("=" * 74)
    print("D: learning rate (seed 1, the one that stalled)")
    print("=" * 74)
    print(f"{'lr':>8s} {'acc@32':>8s}")
    for lr in (1e-3, 3e-3, 1e-2):
        model, head, _ = train_calm(args, batches, autoencoder, cli.steps, lr, 1)
        print(f"{lr:8.0e} {accuracy(model, head, autoencoder, batches, args, 32):7.2%}")


if __name__ == "__main__":
    main()
