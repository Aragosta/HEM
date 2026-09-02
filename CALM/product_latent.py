"""A product-of-Gaussian-manifolds latent, with learnable per-factor curvature.

``RESEARCH.md`` argued that this project has been asking the wrong question.
"Should the latent be hyperbolic or Euclidean" is not HELM's question: HELM's
contribution is a **Mixture of Curvature Experts**, the claim that one curvature
does not fit all of language. The question that follows is *how much* curvature
each latent factor wants, with Euclidean recovered as a limit rather than
assumed or rejected. This module implements the latent that can answer it.

**The construction** (transcribed from ``gmvae/upstream/distributions/PGMNormal``
and its ``LearnablePGMNormal`` sibling). A point in the latent is a pair
``(alpha, log beta^2)`` -- the parameters of a univariate Gaussian. Under the
Fisher-Rao metric the family of univariate Gaussians is isometric to the
hyperbolic plane, so **a point is a distribution**, and a latent of ``dim``
factors is a product of ``dim`` copies of H^2.

**Why this and not the wrapped normal.** ``hyperbolic_latent.py`` builds a point
by ``expmap(mu, transp0(mu, (0, v)))``, which materialises ``cosh(r)``; past
radius ~8 that leaves float32 (see ``MAX_TANGENT_RADIUS``) and the autoencoder
built on it reconstructs held-out patches at 2.29%. Here:

* no exponential map is ever evaluated -- the coordinates *are* the point;
* the second coordinate is a **logarithm**, so the quantity that overflowed in
  the wrapped normal is already on a log scale and cannot;
* the KL is **closed form** -- a Normal KL plus a Gamma KL -- rather than the
  Monte-Carlo estimator the wrapped normal forces;
* sampling is ``Normal.rsample`` plus a Gamma draw, both standard and
  reparameterised.

**Curvature.** ``c`` is negative; ``c -> 0`` flattens the factor toward Euclidean.
:class:`ProductGaussianLatent` can hold it fixed (GM-VAE's ``PGMNormal``) or
learn one value per factor (``LearnablePGMNormal``), which is the setting that
makes the curvature question measurable instead of assumed.

**Width accounting.** Each factor carries two numbers, so ``dim`` factors is
``2 * dim`` values reaching the decoder. A comparison against a Euclidean latent
of width ``L`` must therefore use ``dim = L // 2``, or it is a capacity
comparison wearing a geometry label. :class:`ProductPatchAutoencoder` takes
``latent_size`` in *values* and halves it, so the interface matches the other
autoencoders and the comparison stays honest.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["ProductGaussianNormal", "ProductPatchAutoencoder",
           "fisher_rao_distance", "euclidean_kl_div", "gamma_kl_div"]


# ------------------------------------------------------- closed-form KL terms

def euclidean_kl_div(mean1, logvar1, mean2, logvar2):
    """KL between two univariate normals, elementwise. GM-VAE's ``utils.py``."""
    kl = logvar2 - logvar1
    kl = kl + (logvar1 - logvar2).exp()
    kl = kl + (mean1 - mean2).pow(2) / logvar2.exp()
    return 0.5 * (kl - 1)


def gamma_kl_div(a1, logb1, a2, logb2):
    """KL between two gammas, elementwise. GM-VAE's ``utils.py``."""
    kl = a2 * (logb1 - logb2)
    kl = kl - (torch.lgamma(a1) - torch.lgamma(a2))
    kl = kl + (a1 - a2) * torch.digamma(a1)
    return kl - (1 - (logb2 - logb1).exp()) * a1


# ------------------------------------------------------------------- geometry

def fisher_rao_distance(a: torch.Tensor, b: torch.Tensor,
                        c: float | torch.Tensor = -1.0) -> torch.Tensor:
    """Hyperbolic distance between Gaussians, summed over factors.

    Points are ``(..., dim, 2)`` holding ``(mean, log variance)``. In upper
    half-plane coordinates ``z = (sqrt(2) * mean, std)`` the Fisher-Rao metric of
    the univariate normal family is the hyperbolic metric, and

        d(z1, z2) = arccosh(1 + |z1 - z2|^2 / (2 y1 y2))

    Written through ``arccosh(1 + u)`` with ``u`` formed as a ratio of
    differences, so the small-distance regime does not lose precision the way
    ``arccosh(-<x,y>)`` does on the hyperboloid.
    """
    scale = math.sqrt(2.0)
    x1, y1 = scale * a[..., 0], (0.5 * a[..., 1]).exp()
    x2, y2 = scale * b[..., 0], (0.5 * b[..., 1]).exp()
    numerator = (x1 - x2).square() + (y1 - y2).square()
    u = numerator / (2 * y1 * y2).clamp_min(1e-12)
    # arccosh(1 + u) = 2 asinh(sqrt(u / 2)), stable as u -> 0.
    per_factor = 2 * torch.asinh((u / 2).clamp_min(0).sqrt())
    curvature = torch.as_tensor(c, dtype=a.dtype, device=a.device).abs()
    return (per_factor / curvature.sqrt()).sum(-1)


# --------------------------------------------------------------- distribution

class ProductGaussianNormal:
    """GM-VAE's ``PGMNormal``: a distribution over points of ``(H^2)^dim``.

    Args:
        means: ``(..., dim, 2)`` holding ``(alpha, log beta^2)`` per factor.
        log_gamma_square: ``(..., dim)`` spread of the distribution *over* those
            points -- the variational width, distinct from ``log beta^2``, which
            is a coordinate of the point itself.
        c: curvature, negative. Scalar, or ``(..., dim)`` to vary per factor.
    """

    def __init__(self, means: torch.Tensor, log_gamma_square: torch.Tensor,
                 c: float | torch.Tensor = -1.0):
        self.c = torch.as_tensor(c, dtype=means.dtype, device=means.device)
        self.alpha = means[..., 0]
        self.log_beta_square = means[..., 1]
        self.log_gamma_square = log_gamma_square

        self.normal_mu = self.alpha
        self.normal_logvar = self.log_beta_square + self.log_gamma_square
        four_k = 4 * (-self.c)
        self.gamma_a = (-self.log_gamma_square).exp() / four_k + 1
        self.log_gamma_b = -self.normal_logvar - four_k.log()

    def rsample(self, n: Optional[int] = None) -> torch.Tensor:
        """``(n, ..., dim, 2)`` reparameterised draws (``(..., dim, 2)`` if ``n`` is None)."""
        count = 1 if n is None else n
        shape = torch.Size([count]) + self.gamma_a.shape
        std = (0.5 * self.normal_logvar).exp()
        mean = self.normal_mu.expand(shape) + torch.randn(
            shape, dtype=std.dtype, device=std.device) * std.expand(shape)
        # Gamma is reparameterised in PyTorch via _standard_gamma's implicit
        # gradient, so this stays differentiable in gamma_a.
        logvar = (torch._standard_gamma(self.gamma_a.expand(shape))
                  .clamp_min(1e-12).log() - self.log_gamma_b)
        out = torch.stack([mean, logvar], dim=-1)
        return out.squeeze(0) if n is None else out

    def sample(self, n: Optional[int] = None) -> torch.Tensor:
        with torch.no_grad():
            return self.rsample(n)

    @property
    def mode(self) -> torch.Tensor:
        """The point the distribution is centred on -- the analogue of ``mean``."""
        return torch.stack([self.alpha, self.log_beta_square], dim=-1)

    def kl_div(self, other: "ProductGaussianNormal") -> torch.Tensor:
        """Closed form. No Monte Carlo anywhere. ``(..., dim)``."""
        return (euclidean_kl_div(self.normal_mu, self.normal_logvar,
                                 other.normal_mu, other.normal_logvar)
                + gamma_kl_div(self.gamma_a, self.log_gamma_b,
                               other.gamma_a, other.log_gamma_b))

    @classmethod
    def standard(cls, like: "ProductGaussianNormal") -> "ProductGaussianNormal":
        """The prior GM-VAE uses: zeros for both the point and the spread."""
        means = torch.zeros((*like.alpha.shape, 2), dtype=like.alpha.dtype,
                            device=like.alpha.device)
        return cls(means, torch.zeros_like(like.log_gamma_square), like.c)


# ---------------------------------------------------------------- autoencoder

class ProductPatchAutoencoder(nn.Module):
    """:class:`~CALM.helm_calm.PatchAutoencoder` with a product-manifold latent.

    Same blocks, widths and tied head as the Euclidean and wrapped-normal
    versions, so a comparison between the three isolates the latent. ``dim``
    factors carry ``2 * dim`` values, so ``latent_size`` is halved to keep the
    number of values reaching the decoder equal across all three.

    Args:
        learnable_curvature: when true, the encoder emits a per-factor ``log|c|``
            (GM-VAE's ``LearnablePGMNormal``), so each factor learns how curved
            it wants to be and Euclidean is available as a limit rather than a
            separate model.
    """

    is_product = True

    def __init__(self, vocab_size: int, hidden: int = 256, latent_size: int = 128,
                 patch_size: int = 4, layers: int = 2, c: float = -1.0,
                 learnable_curvature: bool = False,
                 kl_clamp: float = 0.5, dropout: float = 0.15,
                 log_clamp: float = 6.0):
        super().__init__()
        from CALM.helm_calm import PatchAutoencoder  # same Block, no duplication

        if latent_size % 2:
            raise ValueError("latent_size counts values, and each factor holds "
                             f"two, so it must be even; got {latent_size}")
        Block = PatchAutoencoder.Block
        self.patch_size = patch_size
        self.latent_size = latent_size
        self.factors = latent_size // 2
        self.c = c
        self.learnable_curvature = learnable_curvature
        self.kl_clamp = kl_clamp
        self.dropout = dropout
        # Bound on log beta^2 and log gamma^2. It exists to keep the Gamma shape
        # parameter where lgamma and digamma are well conditioned, not to
        # constrain the model -- so it has to be checked for bindingness rather
        # than assumed harmless. At 6.0 it was pinning roughly half of every
        # batch, which makes an accuracy measured under it partly a measurement
        # of the bound. `latent_geometry.py` reports the pinned fraction.
        self.log_clamp = log_clamp

        self.embed = nn.Embedding(vocab_size, hidden)
        self.enc_a = nn.ModuleList([Block(hidden) for _ in range(layers)])
        self.squeeze = nn.Linear(patch_size * hidden, hidden)
        self.enc_b = nn.ModuleList([Block(hidden) for _ in range(layers)])
        self.enc_norm = nn.RMSNorm(hidden, eps=1e-5)
        # alpha, log beta^2, log gamma^2 -- plus log|c| when curvature is learned.
        self.to_latent = nn.Linear(hidden, self.factors * (4 if learnable_curvature else 3))
        self.from_latent = nn.Linear(latent_size, hidden)
        self.dec_a = nn.ModuleList([Block(hidden) for _ in range(layers)])
        self.expand = nn.Linear(hidden, patch_size * hidden)
        self.dec_b = nn.ModuleList([Block(hidden) for _ in range(layers)])
        self.dec_norm = nn.RMSNorm(hidden, eps=1e-5)
        self.head = nn.Linear(hidden, vocab_size, bias=False)
        self.head.weight = self.embed.weight

    def encode(self, ids: torch.Tensor) -> ProductGaussianNormal:
        h = self.embed(ids)
        for block in self.enc_a:
            h = block(h)
        h = self.squeeze(h.flatten(-2))
        for block in self.enc_b:
            h = block(h)
        parts = self.to_latent(self.enc_norm(h)).chunk(
            4 if self.learnable_curvature else 3, dim=-1)
        alpha, log_beta_sq, log_gamma_sq = parts[:3]
        # Both are logs of variances; clamping keeps the Gamma shape parameter
        # in a range where lgamma and digamma are well conditioned.
        bound = self.log_clamp
        log_beta_sq = log_beta_sq.clamp(-bound, bound)
        log_gamma_sq = log_gamma_sq.clamp(-bound, bound)
        curvature = (-parts[3].clamp(-4.0, 4.0).exp() if self.learnable_curvature
                     else self.c)
        return ProductGaussianNormal(
            torch.stack([alpha, log_beta_sq], dim=-1), log_gamma_sq, curvature)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """``(..., dim, 2)`` points -> ``(..., K, vocab)``."""
        flat = latent.reshape(*latent.shape[:-2], self.latent_size)
        h = self.from_latent(flat)
        for block in self.dec_a:
            h = block(h)
        h = self.expand(h).view(*flat.shape[:-1], self.patch_size, -1)
        for block in self.dec_b:
            h = block(h)
        return self.head(self.dec_norm(h))

    def elbo(self, ids: torch.Tensor, kl_weight: float = 1e-3):
        posterior = self.encode(ids)
        latent = posterior.rsample()
        latent = F.dropout(latent, p=self.dropout, training=self.training)
        logits = self.decode(latent)
        recon = F.cross_entropy(logits.reshape(-1, logits.size(-1)), ids.reshape(-1))
        kl = posterior.kl_div(ProductGaussianNormal.standard(posterior))
        kl = kl.clamp(min=self.kl_clamp).sum(-1).mean()
        return recon * self.patch_size + kl_weight * kl, recon

    def freeze(self) -> "ProductPatchAutoencoder":
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self.eval()
