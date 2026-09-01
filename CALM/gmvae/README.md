# GM-VAE — Hyperbolic VAE via Latent Gaussian Distributions

[arXiv:2209.15217](https://arxiv.org/abs/2209.15217) (Cho, Lee & Kim, NeurIPS
2023) · [github.com/ml-postech/GM-VAE](https://github.com/ml-postech/GM-VAE)

Evaluated here as a candidate for the **hyperbolic latent** in a CALM-HELM —
Stage 3 of `../ASSESSMENT.md`, and as a possible answer to the training
instability Stage 1 found.

```
upstream/     the GM-VAE reference implementation, vendored
ASSESSMENT.md whether it helps CALM-HELM, and what is broken in the release
```

## The idea

The set of univariate Gaussians `N(μ, σ²)`, equipped with the Fisher information
metric, **is** a hyperbolic space — the "Gaussian manifold". So a latent point
can be represented as a `(μ, log σ²)` pair and manipulated hyperbolically without
ever calling `expmap`, `logmap` or `arcosh`.

GM-VAE builds a VAE on that space using a **pseudo-Gaussian manifold normal**
(`PGMNormal`) whose log-density is written in terms of a Gaussian KL divergence —
a local approximation of the squared Fisher–Rao distance. Its headline claim is
the one that matters here:

> "We observe that our model provides strong numerical stability, addressing a
> common limitation reported in previous hyperbolic-VAEs."

## Why it is relevant to CALM-HELM

CALM's autoencoder already emits exactly `(mean, log_std)` per latent dimension,
and its energy score samples targets as `mean + eps * std`. GM-VAE says that same
pair of numbers *is* a point in H². So a `latent_size=128` CALM latent, read as
64 `(μ, log σ²)` pairs, is already a point in a product of hyperbolic planes —
**a hyperbolic latent needs no change to the autoencoder's output shape**, only a
change of distance.

The upstream attribution and citation are in `upstream/README.md`.
