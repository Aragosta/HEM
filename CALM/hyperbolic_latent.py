"""A hyperbolic latent for HELM-CALM: wrapped-normal VAE + geodesic energy score.

``DID_IT_WORK.md`` §5 established, over five seeds, that HELM-CALM sits 2.94
points (4.6 standard errors) behind a Euclidean control and carries roughly
twelve times its seed variance. It also located the only geometric seam left
after the ``LorentzEnergyHead`` fix: **the autoencoder's latent is Euclidean.**

That seam is not cosmetic. It forces two things:

1. ``LorentzEnergyHead.forward`` ends in ``return_space=True`` -- the head is
   pushed off the manifold at its last layer, not by choice but because the
   target it must match lives in a flat space;
2. the energy score ``E d(X,y)^b - 1/2 E d(X,X')^b`` is evaluated with a
   Euclidean ``d`` over a quantity every earlier layer produced hyperbolically.

This module closes the seam. It supplies

* :class:`WrappedNormal` -- the hyperbolic Gaussian of Nagano et al. (2019),
  which is what a VAE posterior on the hyperboloid actually is;
* :func:`lorentz_distance` -- the geodesic distance, in a form that does not
  lose precision near zero (see below);
* :func:`lorentz_energy_score` -- CALM's estimator with that distance;
* :class:`LorentzPatchAutoencoder` -- a drop-in for
  :class:`~CALM.helm_calm.PatchAutoencoder` whose posterior is a point and a
  spread *on the manifold*.

**Why the energy score still works here.** CALM's objective is proper because
``d^b`` is a conditionally negative definite kernel on the sample space for
``b`` in (0, 2) -- true of Euclidean space. Real hyperbolic space is a space of
negative type (Faraut & Harzallah, 1974), so its geodesic distance is
conditionally negative definite and ``d^b`` remains so for ``b`` in **(0, 1]**;
the range does *not* extend to 2 as it does in the Euclidean case. Combined with
the existing constraint that ``b < 1`` gives unbounded self-distance gradients,
this pins ``b = 1`` exactly on the manifold. :func:`lorentz_energy_score`
enforces that rather than leaving it to the caller.

**Why the distance is written the way it is.** The textbook form
``d = sqrt(c) * arccosh(-<x,y>_L / c)`` evaluates ``arccosh`` at ``1 + eps`` for
nearby points, where it loses roughly half the mantissa. For points *on* the
hyperboloid, ``<x-y, x-y>_L = -2(c + <x,y>_L)`` exactly, so the small quantity
can be formed as a difference instead of a cancellation, and

    d = 2 sqrt(c) asinh( sqrt(<x-y, x-y>_L) / (2 sqrt(c)) )

is accurate all the way down to coincident points. Verified against
``geoopt.Lorentz.dist`` to double-precision agreement.

Note in passing that ``helm.hypercore.manifolds.Lorentz.induced_distance`` has a
sign error -- it clamps ``<x,y>_L / c``, which is ``-1`` on the manifold, so it
returns ~0 for every pair. Nothing in HELM's own forward path calls it; this
module does not use it.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from helm.hypercore.manifolds import Lorentz
from helm.hypercore.nn.linear.lorentz_linear import LorentzLinear

__all__ = ["lorentz_distance", "lorentz_energy_score", "WrappedNormal",
           "LorentzPatchAutoencoder"]

_EPS = 1e-12


# ------------------------------------------------------------------- geometry

def lorentz_distance(x: torch.Tensor, y: torch.Tensor, c: float | torch.Tensor = 1.0
                     ) -> torch.Tensor:
    """Geodesic distance between points on the hyperboloid of curvature ``-1/c``.

    Broadcasting, and differentiable at coincident points: the gradient there is
    set to zero, matching what ``torch.linalg.norm`` does at the origin (and what
    CALM's Euclidean estimator therefore relies on for its diagonal terms).

    Args:
        x, y: ``(..., d + 1)`` Lorentz vectors, time coordinate first.
        c: curvature scale.

    Returns:
        ``(...)`` distances.
    """
    diff = x - y
    quad = (diff[..., 1:].square().sum(-1) - diff[..., 0].square())
    sqrt_c = torch.as_tensor(c, dtype=x.dtype, device=x.device).sqrt()
    # clamp_min(_EPS), not clamp_min(0): sqrt has an infinite derivative at 0, so
    # a hard zero would poison the whole batch through torch.where's false branch.
    safe = quad.clamp_min(_EPS)
    dist = 2 * sqrt_c * torch.asinh(safe.sqrt() / (2 * sqrt_c))
    return torch.where(quad > _EPS, dist, torch.zeros_like(dist))


def lorentz_energy_score(samples: torch.Tensor, posterior: "WrappedNormal",
                         n_target: int = 100, c: float | torch.Tensor = 1.0
                         ) -> torch.Tensor:
    """CALM's energy score with the geodesic distance in place of the L2 norm.

    Mirrors :func:`CALM.helm_calm.energy_score` term for term -- same estimator,
    same normalisation ``n(n-1)`` for the pairwise term, same factor of two on
    the cross term -- so a comparison between the two isolates the geometry and
    nothing else.

    ``beta`` is fixed at 1: see the module docstring for why the hyperbolic case
    admits no other value.

    Args:
        samples: ``(n_samples, tokens, d + 1)`` draws from the head, on manifold.
        posterior: the target distribution, sampled ``n_target`` times.
        n_target: Monte-Carlo draws from the target.
        c: curvature scale.

    Returns:
        ``(tokens,)`` score; the training loss is its negation.
    """
    n_samples = samples.shape[0]
    if n_samples < 2:
        raise ValueError("the pairwise term needs at least two samples")

    pairwise = lorentz_distance(samples.unsqueeze(1), samples.unsqueeze(0), c)
    distance_x = pairwise.sum(dim=(0, 1)) / (n_samples * (n_samples - 1))

    targets = posterior.sample(n_target)
    cross = lorentz_distance(samples.reshape(n_samples, 1, *samples.shape[1:]),
                             targets.reshape(1, n_target, *targets.shape[1:]), c)
    return distance_x - cross.mean(dim=(0, 1)) * 2


class WrappedNormal:
    """The wrapped normal on the Lorentz hyperboloid (Nagano et al., 2019).

    A Euclidean Gaussian has no canonical analogue on a manifold; the wrapped
    normal is the construction that keeps a closed-form density and a
    reparameterised sample. Draw ``v ~ N(0, diag(sigma^2))`` in ``R^d``, read it
    as a tangent vector ``(0, v)`` at the origin, parallel-transport it to the
    mean, and exponentiate:

        z = exp_mu( PT_{o -> mu}( (0, v) ) )

    The change of variables is radial, so the density is the Euclidean one times
    a single Jacobian factor:

        log p(z) = log N(v; 0, sigma) - (d - 1) log( sinh(r) / r ),  r = ||v||

    This is what makes the class usable as a VAE posterior: the ELBO needs a
    density, and the energy score needs only samples.

    Args:
        manifold: the :class:`~helm.hypercore.manifolds.Lorentz` to live on.
        mean: ``(..., d + 1)`` points on the manifold.
        log_std: ``(..., d)`` per-coordinate log standard deviations, in the
            tangent space -- one fewer than ``mean`` because the tangent space at
            a point of the ``d``-dimensional hyperboloid is ``d``-dimensional.
    """

    def __init__(self, manifold: Lorentz, mean: torch.Tensor, log_std: torch.Tensor):
        if mean.shape[-1] != log_std.shape[-1] + 1:
            raise ValueError(f"mean has {mean.shape[-1]} coordinates, so log_std "
                             f"should have {mean.shape[-1] - 1}, got {log_std.shape[-1]}")
        self.manifold = manifold
        self.mean = mean
        self.log_std = log_std
        self.dim = log_std.shape[-1]

    def _lift(self, v: torch.Tensor) -> torch.Tensor:
        """``(..., d)`` -> a tangent vector at the origin, ``(..., d + 1)``."""
        return torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)

    def rsample(self, n: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reparameterised draw. Returns ``(z, v)``; ``v`` is the tangent noise."""
        shape = self.mean.shape if n is None else (n, *self.mean.shape)
        v = torch.randn((*shape[:-1], self.dim), dtype=self.mean.dtype,
                        device=self.mean.device) * self.log_std.exp()
        mean = self.mean if n is None else self.mean.expand(shape)
        u = self.manifold.transp0(mean, self._lift(v))
        return self.manifold.expmap(mean, u), v

    def sample(self, n: Optional[int] = None) -> torch.Tensor:
        """Draw without keeping the tangent noise."""
        return self.rsample(n)[0]

    def log_prob(self, v: torch.Tensor) -> torch.Tensor:
        """Density of the sample whose tangent noise was ``v``.

        Taking ``v`` rather than ``z`` avoids inverting the transport, which is
        what the reparameterised ELBO wants anyway.
        """
        log_std = self.log_std.expand_as(v)
        gaussian = (-0.5 * (v / log_std.exp()).square()
                    - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        radius = v.square().sum(-1).clamp_min(_EPS).sqrt()
        # log(sinh(r)/r), stable at r -> 0 where the ratio tends to 1.
        log_ratio = torch.where(
            radius > 1e-4,
            torch.log(torch.sinh(radius.clamp(max=30.0)) / radius.clamp_min(_EPS)),
            radius.square() / 6.0)
        # Large r: sinh(r)/r overflows well before r = 30, so use the asymptote.
        log_ratio = torch.where(radius > 30.0,
                                radius - math.log(2.0) - radius.log(), log_ratio)
        return gaussian - (self.dim - 1) * log_ratio

    def kl_to_origin_prior(self, n: int = 1) -> torch.Tensor:
        """Monte-Carlo ``KL(q || p)`` against a unit wrapped normal at the origin.

        The wrapped normal has no closed-form KL, so this is the single-sample
        estimator every hyperbolic VAE uses. It is unbiased and shares the
        reparameterised sample with the reconstruction term.
        """
        z, v = self.rsample(n)
        origin = torch.zeros_like(self.mean).expand(z.shape).contiguous()
        origin[..., 0] = torch.as_tensor(self.manifold.c, device=z.device).sqrt()
        prior = WrappedNormal(self.manifold, origin,
                              torch.zeros_like(self.log_std).expand(z.shape[:-1]
                                                                   + (self.dim,)))
        # The prior's density at z needs z's own tangent coordinates at the
        # origin, which for a mean at the origin is just logmap0's space part.
        v_prior = self.manifold.logmap0(z)[..., 1:]
        return (self.log_prob(v) - prior.log_prob(v_prior)).mean(0)


# ---------------------------------------------------------------- autoencoder

class LorentzPatchAutoencoder(nn.Module):
    """:class:`~CALM.helm_calm.PatchAutoencoder` with the latent on a hyperboloid.

    Deliberately a *minimal* change from the Euclidean version: same block
    structure, same widths, same tied head, same ``patch_size`` and
    ``latent_size`` semantics. What differs is confined to the latent:

    ================  ===========================  ==============================
    piece             Euclidean                    here
    ================  ===========================  ==============================
    posterior         ``N(mu, sigma)`` in R^L      wrapped normal on H^L
    ``encode``        ``(mean, log_std)``          :class:`WrappedNormal`
    decoder entry     ``nn.Linear``                ``LorentzLinear``
    KL                closed form                  1-sample Monte Carlo
    ================  ===========================  ==============================

    The encoder and decoder *bodies* stay Euclidean on purpose. They consume and
    produce token embeddings, not manifold points, so there is nothing hyperbolic
    about them to preserve -- making them hyperbolic too would confound the
    experiment this class exists to run.

    ``encode`` returns a distribution object rather than a ``(mean, log_std)``
    pair, which is the one interface difference callers must handle;
    :class:`~CALM.helm_calm.HelmCALM` branches on ``latent_geometry`` for it.
    """

    is_hyperbolic = True

    def __init__(self, vocab_size: int, hidden: int = 256, latent_size: int = 128,
                 patch_size: int = 4, layers: int = 2,
                 manifold: Optional[Lorentz] = None):
        super().__init__()
        from CALM.helm_calm import PatchAutoencoder  # same Block, no duplication

        self.manifold = manifold if manifold is not None else Lorentz(1.0)
        self.patch_size = patch_size
        self.latent_size = latent_size
        Block = PatchAutoencoder.Block

        self.embed = nn.Embedding(vocab_size, hidden)
        self.enc_a = nn.ModuleList([Block(hidden) for _ in range(layers)])
        self.squeeze = nn.Linear(patch_size * hidden, hidden)
        self.enc_b = nn.ModuleList([Block(hidden) for _ in range(layers)])
        self.enc_norm = nn.RMSNorm(hidden, eps=1e-5)
        # The mean is produced in the tangent space at the origin and pushed onto
        # the manifold; the spread stays a tangent quantity throughout.
        self.to_mean = nn.Linear(hidden, latent_size)
        self.to_log_std = nn.Linear(hidden, latent_size)
        self.from_latent = LorentzLinear(self.manifold, latent_size + 1, hidden)
        self.dec_a = nn.ModuleList([Block(hidden) for _ in range(layers)])
        self.expand = nn.Linear(hidden, patch_size * hidden)
        self.dec_b = nn.ModuleList([Block(hidden) for _ in range(layers)])
        self.dec_norm = nn.RMSNorm(hidden, eps=1e-5)
        self.head = nn.Linear(hidden, vocab_size, bias=False)
        self.head.weight = self.embed.weight
        # Start the posterior tight and near the origin, as the Euclidean version
        # effectively does; a diffuse init on a hyperboloid puts mass at radii
        # where sinh(r) is already enormous.
        nn.init.zeros_(self.to_mean.weight)
        nn.init.zeros_(self.to_mean.bias)
        nn.init.zeros_(self.to_log_std.weight)
        nn.init.constant_(self.to_log_std.bias, -2.0)

    def encode(self, ids: torch.Tensor) -> WrappedNormal:
        """``(N, K)`` ids -> the posterior over the patch's latent."""
        h = self.embed(ids)
        for block in self.enc_a:
            h = block(h)
        h = self.squeeze(h.flatten(-2))
        for block in self.enc_b:
            h = block(h)
        h = self.enc_norm(h)
        tangent = self.to_mean(h)
        mean = self.manifold.expmap0(
            torch.cat([torch.zeros_like(tangent[..., :1]), tangent], dim=-1))
        return WrappedNormal(self.manifold, mean,
                             self.to_log_std(h).clamp(-6.0, 2.0))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """``(..., L + 1)`` manifold points -> ``(..., K, vocab)``."""
        h = self.from_latent(latent, return_space=True)
        for block in self.dec_a:
            h = block(h)
        h = self.expand(h).view(*latent.shape[:-1], self.patch_size, -1)
        for block in self.dec_b:
            h = block(h)
        return self.head(self.dec_norm(h))

    def elbo(self, ids: torch.Tensor, kl_weight: float = 1e-3):
        """Reconstruction + Monte-Carlo KL. Returns ``(loss, recon_ce)``."""
        posterior = self.encode(ids)
        latent, _ = posterior.rsample()
        logits = self.decode(latent)
        recon = F.cross_entropy(logits.reshape(-1, logits.size(-1)), ids.reshape(-1))
        kl = posterior.kl_to_origin_prior().mean()
        return recon * self.patch_size + kl_weight * kl, recon

    def freeze(self) -> "LorentzPatchAutoencoder":
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self.eval()
