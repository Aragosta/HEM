#!/usr/bin/env python3
"""Can the autoencoder meet CALM's precondition? A go/no-go test.

CALM's README states the requirement plainly: the autoencoder must "compress K
tokens into a single vector and reconstruct them with **near-perfect
accuracy**". That is not a nice-to-have. The pipeline is

    backbone -> latent -> autoencoder.decode -> K tokens

so **the autoencoder's round-trip accuracy is a hard ceiling on the entire
model**. A model whose autoencoder reconstructs at 60% cannot score above 60%
however good its backbone is.

This makes for a cheap decision procedure that does not require training a
backbone at all:

* if a small autoencoder *can* reach near-lossless reconstruction here, then
  CALM's precondition is met and any subsequent quality gap is attributable to
  the backbone, the objective, or the geometry;
* if it *cannot*, then no result from this repository is informative about CALM
  in either direction -- the failure would be ours, and CALM's mechanism would
  never have been given a chance to be wrong.

CALM meets the precondition with a 75.8M-parameter autoencoder trained on real
corpora. This asks what is reachable at toy scale on 850KB of bytes, sweeping
the two knobs that matter: latent width (how much room the vector has) and
hidden width (how much capacity the encoder has to use it).

The Euclidean and hyperbolic latents are both swept, because the wrapped
normal's density carries a factor of ``(d - 1)`` -- at latent width 32 the KL
penalises radius 31x -- which predicts the hyperbolic variant should *improve*
as the latent gets narrower, the opposite of the Euclidean one. That prediction
is recorded here before the numbers are in.

Usage: python CALM/experiments/ae_fidelity.py --steps 4000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
for _extra in (ROOT, ROOT / "CALM", ROOT / "CALM" / "experiments"):
    sys.path.insert(0, str(_extra))

from helm_calm import PatchAutoencoder  # noqa: E402
from hyperbolic_latent import LorentzPatchAutoencoder  # noqa: E402
from real_text import VOCAB, build_corpus  # noqa: E402


def patches(data: torch.Tensor, patch: int) -> torch.Tensor:
    usable = (data.numel() // patch) * patch
    return data[:usable].view(-1, patch)


def fit(cls, train, patch, latent, hidden, steps, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    model = cls(VOCAB, hidden=hidden, latent_size=latent, patch_size=patch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    rows = patches(train, patch)
    for _ in range(steps):
        index = torch.randint(0, rows.size(0), (256,), generator=generator)
        loss, _ = model.elbo(rows[index])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model.eval()


@torch.no_grad()
def fidelity(model, data, patch) -> tuple:
    """Per-byte and per-patch reconstruction accuracy.

    Per-*patch* is the one that matters: CALM decodes a whole patch from one
    vector, so a patch with one wrong byte is a wrong patch.
    """
    rows = patches(data, patch)[:8192]
    posterior = model.encode(rows)
    latent = (posterior.mean if getattr(model, "is_hyperbolic", False)
              else posterior[0])
    predicted = model.decode(latent).argmax(-1)
    per_byte = (predicted == rows).float().mean().item()
    per_patch = (predicted == rows).all(dim=-1).float().mean().item()
    return per_byte, per_patch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--patch", type=int, default=4)
    parser.add_argument("--latents", default="8,16,32,64")
    parser.add_argument("--hidden", default="128,256")
    options = parser.parse_args()

    train, valid = build_corpus()
    print(f"K = {options.patch}, {options.steps} steps, "
          f"{train.numel():,} train bytes / {valid.numel():,} held out")
    print("CALM's precondition is near-perfect reconstruction; per-patch is the "
          "number that\nmatters, since a patch with one wrong byte is a wrong "
          "patch.\n")
    print(f"{'latent':>7s} {'hidden':>7s} {'geometry':>11s} "
          f"{'byte (train)':>13s} {'byte (held)':>12s} {'patch (held)':>13s}")

    for hidden in (int(v) for v in options.hidden.split(",")):
        for latent in (int(v) for v in options.latents.split(",")):
            for name, cls in (("euclidean", PatchAutoencoder),
                              ("hyperbolic", LorentzPatchAutoencoder)):
                model = fit(cls, train, options.patch, latent, hidden,
                            options.steps)
                train_byte, _ = fidelity(model, train, options.patch)
                held_byte, held_patch = fidelity(model, valid, options.patch)
                print(f"{latent:7d} {hidden:7d} {name:>11s} "
                      f"{train_byte:13.2%} {held_byte:12.2%} {held_patch:13.2%}",
                      flush=True)


if __name__ == "__main__":
    main()
