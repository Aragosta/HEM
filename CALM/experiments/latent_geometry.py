#!/usr/bin/env python3
"""Which latent geometry works for CALM, and *why* — measured, not argued.

Four latents on the identical autoencoder -- same blocks, widths, tied head,
data, budget and optimizer -- so the only variable is the latent:

``euclidean``    CALM's own: a diagonal Gaussian with a clamped KL.
``wrapped``      wrapped normal on the Lorentz hyperboloid (Nagano et al.), the
                 construction ``hyperbolic_latent.py`` implements.
``product``      GM-VAE's product of Gaussian manifolds: a point is a pair
                 ``(alpha, log beta^2)``, which under the Fisher-Rao metric is a
                 point of H^2. Curvature fixed.
``learnable``    the same with per-factor learnable curvature, so Euclidean is
                 recovered as ``c -> 0`` rather than assumed or rejected.

**Matched width.** A product factor carries two values, so ``latent_size`` counts
*values* everywhere and the product latents use half as many factors. Otherwise
this is a capacity comparison wearing a geometry label.

**The metric.** Held-out per-patch reconstruction: CALM's stated precondition is
that the autoencoder reconstructs K tokens "with near-perfect accuracy", and it
is a hard ceiling on any model built on the latent. A patch with one wrong byte
is a wrong patch.

**The instrumentation is the point.** Reporting only accuracy would repeat this
project's recurring mistake -- reading a broken component as an architectural
result. The wrapped normal's 2.29% was a float32 failure, not a verdict on
geometry, and only the radius trace showed that. So every run also reports:

``radius``      how far the latent sits from the origin, in the units each
                parameterisation uses. For the wrapped normal this is the
                quantity that hit the float32 cliff at ~8; for the product
                latent it is a Fisher-Rao distance that cannot blow up the same
                way, because the coordinate is already a logarithm.
``clamped``     fraction of the batch pinned against a clamp. Anything near 1
                means the model wants room the representation will not give it,
                and the accuracy below it is a measurement of the clamp.
``KL``          nats actually used. A collapsed KL means the decoder is ignoring
                the latent and the accuracy is not about the latent at all.
``nonfinite``   steps that produced a NaN or Inf loss. Any value above zero
                invalidates that row.

Usage::

    python CALM/experiments/latent_geometry.py --steps 2000
    python CALM/experiments/latent_geometry.py --steps 2000 --latents 16,64
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
for _extra in (ROOT, ROOT / "CALM", ROOT / "CALM" / "experiments"):
    sys.path.insert(0, str(_extra))

from helm.hypercore.manifolds import Lorentz  # noqa: E402
from helm_calm import PatchAutoencoder  # noqa: E402
from hyperbolic_latent import MAX_TANGENT_RADIUS, LorentzPatchAutoencoder  # noqa: E402
from product_latent import ProductPatchAutoencoder, fisher_rao_distance  # noqa: E402
from real_text import VOCAB, build_corpus  # noqa: E402

BUILDERS = {
    "euclidean": lambda **kw: PatchAutoencoder(**kw),
    "wrapped": lambda **kw: LorentzPatchAutoencoder(**kw),
    "product": lambda **kw: ProductPatchAutoencoder(**kw),
    "learnable": lambda **kw: ProductPatchAutoencoder(learnable_curvature=True, **kw),
}


def patches(data: torch.Tensor, patch: int) -> torch.Tensor:
    return data[:(data.numel() // patch) * patch].view(-1, patch)


def latent_point(model, posterior):
    """The point the posterior is centred on, whatever the parameterisation."""
    if getattr(model, "is_product", False):
        return posterior.mode
    if getattr(model, "is_hyperbolic", False):
        return posterior.mean
    return posterior[0]


@torch.no_grad()
def probe(model, posterior):
    """Radius, clamp pressure and KL, in each parameterisation's own units."""
    point = latent_point(model, posterior)
    if getattr(model, "is_product", False):
        origin = torch.zeros_like(point)
        radius = fisher_rao_distance(point, origin, posterior.c)
        clamped = ((posterior.log_beta_square.abs() > 5.99).float().mean()
                   + (posterior.log_gamma_square.abs() > 5.99).float().mean()) / 2
        kl = posterior.kl_div(type(posterior).standard(posterior)).sum(-1)
    elif getattr(model, "is_hyperbolic", False):
        radius = model.manifold.logmap0(point)[..., 1:].norm(dim=-1)
        clamped = (radius > MAX_TANGENT_RADIUS - 1e-3).float().mean()
        kl = posterior.kl_to_origin_prior()
    else:
        mean, log_std = posterior
        radius = mean.norm(dim=-1)
        clamped = torch.zeros(())
        kl = (0.5 * (mean.pow(2) + (2 * log_std).exp() - 1) - log_std).sum(-1)
    return radius.mean().item(), radius.max().item(), clamped.item(), kl.mean().item()


def fit(name, train, patch, latent, hidden, steps, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    model = BUILDERS[name](vocab_size=VOCAB, hidden=hidden, latent_size=latent,
                           patch_size=patch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    rows = patches(train, patch)
    nonfinite = 0
    for _ in range(steps):
        index = torch.randint(0, rows.size(0), (256,), generator=generator)
        loss, _ = model.elbo(rows[index])
        if not torch.isfinite(loss):
            nonfinite += 1
            optimizer.zero_grad()
            continue
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model.eval(), nonfinite


@torch.no_grad()
def fidelity(model, data, patch, limit=8192):
    rows = patches(data, patch)[:limit]
    posterior = model.encode(rows)
    predicted = model.decode(latent_point(model, posterior)).argmax(-1)
    hit = predicted == rows
    return hit.float().mean().item(), hit.all(dim=-1).float().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--patch", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--latents", default="16,64")
    parser.add_argument("--which", default="euclidean,wrapped,product,learnable")
    options = parser.parse_args()

    train, valid = build_corpus()
    print(f"K = {options.patch}, hidden = {options.hidden}, {options.steps} steps, "
          f"{train.numel():,} train / {valid.numel():,} held-out bytes")
    print("latent_size counts VALUES: a product factor holds two, so those "
          "latents use half as many\nfactors and every row sees the same number "
          "of numbers reaching the decoder.\n")
    print(f"{'latent':>7s} {'geometry':>10s} {'byte':>7s} {'patch':>7s} "
          f"{'radius':>15s} {'clamped':>8s} {'KL':>9s} {'NaN':>5s}")

    for latent in (int(v) for v in options.latents.split(",")):
        for name in options.which.split(","):
            model, nonfinite = fit(name, train, options.patch, latent,
                                   options.hidden, options.steps)
            byte, patch_accuracy = fidelity(model, valid, options.patch)
            with torch.no_grad():
                posterior = model.encode(patches(valid, options.patch)[:4096])
                mean_r, max_r, clamped, kl = probe(model, posterior)
            flag = "" if nonfinite == 0 else "  <- row invalid"
            print(f"{latent:7d} {name:>10s} {byte:7.2%} {patch_accuracy:7.2%} "
                  f"{mean_r:6.2f} (max {max_r:5.2f}) {clamped:8.1%} {kl:9.2f} "
                  f"{nonfinite:5d}{flag}", flush=True)

    print("\nReading this table:")
    print("  clamped near 100%  the latent is pinned against a representational")
    print("                     limit; the accuracy measures the clamp, not the")
    print("                     geometry")
    print("  KL near 0          posterior collapse: the decoder ignores the")
    print("                     latent, so the accuracy is not about it either")
    print("  NaN above 0        the row is invalid regardless of its accuracy")
    print("  radius             wrapped: tangent radius, float32 fails past ~8.")
    print("                     product: Fisher-Rao distance; its coordinate is")
    print("                     already a logarithm, so it cannot blow up the")
    print("                     same way.")


if __name__ == "__main__":
    main()
