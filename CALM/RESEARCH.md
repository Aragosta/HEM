# Hyperbolic latents, revisited — the failure I measured is a known one, with a known fix

A second pass through the hyperbolic-VAE literature and through the GM-VAE code
we already cloned. It **corrects a conclusion given earlier in this project**:
that a hyperbolic latent is unworkable for a CALM-style model in float32. That
was inferred from the wrapped normal, which turns out to be the fragile member
of the family, and the conclusion does not survive contact with the alternatives.

## 1. What we measured was a documented failure mode, not a discovery

The radius cliff — coordinates growing as `cosh(r)`, the hyperboloid constraint
falling below the float32 noise floor past radius ~8, `expmap`/`transp`
returning NaN — is not news to the field. Surveys of hyperbolic VAEs name
numerical stability as *a common limitation of previous hyperbolic VAEs*, and
locate it specifically in the geodesic distance of hyperbolic space
([Hyperbolic VAE via Latent Gaussian Distributions](https://arxiv.org/pdf/2209.15217)).

So the correct reading of our early wrapped-normal numbers is not "hyperbolic
latents do not work". It is "we picked the parameterisation the literature warns
about, and got the failure the literature predicts."

**Correction to the numbers this document originally quoted.** It cited 2.29%
held-out per-patch reconstruction for the wrapped normal as a fact about our
implementation. It is not. In an otherwise identical run -- same model, data,
budget and radius clamp -- the wrapped normal scores **80.91%**; the two runs
differ only in that the second clips gradient norms and skips non-finite steps.
A figure quoted from a single uncontrolled run is not a property of the method,
and this one was repeated several times before that was noticed. Gradient
clipping is now an explicit variable in `latent_geometry.py` rather than an
incidental convenience.

## 2. GM-VAE removes the failure structurally, not by clamping

The construction (`gmvae/upstream/distributions/PGMNormal/`) is worth stating
precisely, because it is not "a wrapped normal with better constants" — it is a
different object.

**The latent point is a pair `(α, log β²)`: the parameters of a univariate
Gaussian.** Under the Fisher–Rao metric the family of univariate Gaussians is
isometric to the hyperbolic plane, so a point *is* a distribution, and a
`latent_dim`-wide latent is a **product of `latent_dim` copies of H²** — hence
PGM, product of Gaussian manifolds.

Three consequences, each of which removes something that broke in our build:

| our wrapped normal | GM-VAE |
| --- | --- |
| point built by `expmap(μ, transp₀(μ, (0,v)))` — materialises `cosh(r)` | point **is** `(α, log β²)`; no exponential map is ever evaluated |
| radius unbounded, clamped by hand at 5 after reaching 9.46 | the variance coordinate is **stored as a logarithm**, so the quantity that blew up is already on a log scale |
| KL by Monte Carlo (no closed form) | **closed form**: `euclidean_kl_div + gamma_kl_div`, a Normal KL plus a Gamma KL |
| sampling through `expmap`/`transp` | `Normal.rsample` for the mean coordinate, `Gamma` for the variance coordinate — both standard and reparameterised |

The middle row is the one that matters for what we measured. Our failure was
`cosh(r)` overflowing the float32 mantissa. **GM-VAE never forms `cosh(r)`.** The
radius cliff cannot occur in that parameterisation, so the float32 argument
given earlier in this project simply does not apply to it.

The same reasoning favours the **Poincaré ball** over the Lorentz hyperboloid
generally: the ball stores points inside a bounded region rather than out at
`cosh(r)`. Same geometry, radically different conditioning.

Note also from the same literature that the **Riemannian normal** — the
maximum-entropy distribution, which uses geodesic distance in its density — is
*harder* to sample (rejection sampling) and in practice performs similarly to
the wrapped normal, especially in high dimensions. It is not the way out.

## 3. The connection to MiCE that we have been walking past

GM-VAE ships `LearnablePGMNormal`, whose encoder emits a per-factor `logc` —
**learnable curvature, one per latent factor**. And
[Mixed-curvature VAEs](https://arxiv.org/abs/1911.08411) (ICLR 2020) build
latents that are products of constant-curvature manifolds, hyperbolic, Euclidean
and spherical together, with per-component curvature fixed or learned.

That is HELM's own thesis, stated in latent space. HELM's contribution is
**Mixture of Curvature Experts** — the claim that one curvature does not fit all
of language. We have spent this project asking "should the latent be hyperbolic
or Euclidean", which is the wrong question by HELM's own argument. The question
HELM implies is **"how much curvature does each latent factor want, and can it be
learned?"** — and a product latent with per-factor curvature answers it, with
Euclidean recovered as the `c → 0` case rather than assumed or rejected.

Precedent for a hyperbolic latent over text specifically exists too:
[APo-VAE](https://arxiv.org/pdf/2005.00054) does text generation in hyperbolic
space.

## 4. What this changes, and what it does not

**Corrected.** Earlier: *"a hyperbolic VAE should not be the latent of a
CALM-style model in float32"*, argued from the radius budget. That argument is
sound about the wrapped normal on the hyperboloid and does not generalise. GM-VAE
sidesteps it by construction.

**Superseded.** This section originally reported the Euclidean autoencoder at
89.11% against the wrapped normal's 2.29% and called the second a fact about our
implementation. Both figures came from single unclipped runs. Under controlled
conditions -- gradient clipping on, non-finite steps skipped, one seed, latent 16
-- the picture is different and much closer:

| latent | held-out per patch | radius | pinned against a clamp | KL |
| --- | --- | --- | --- | --- |
| euclidean | 87.83% | 10.24 (max 18.08) | 0.0% | 92.34 |
| wrapped normal | 80.91% | 5.00 (max 5.00) | **100.0%** | 68.50 |
| product (GM-VAE) | 88.34% | 18.18 (max 36.59) | 48.7% | 72.76 |
| product, learnable curvature | **89.42%** | 32.35 (max 70.29) | 47.6% | 98.95 |

Read the `pinned` column before the accuracy column. The wrapped normal is
against its radius clamp for the whole batch, and both product latents for half
of theirs, so three of those four accuracies are partly measurements of our own
hyperparameters rather than of geometry. Re-running with non-binding clamps
across seeds is what decides whether the ordering is real.

**The one number that explains the wrapped normal's ceiling.** The Euclidean
latent, which is unconstrained, settles at radius **10.24 and reaches 18.08**.
Float32 on the hyperboloid fails past about **8**. So this task wants more
dynamic range than that parameterisation has, and the wrapped normal's earlier
drift to radius 22 was not pathology -- it was the model reaching for the range
it needed and falling off the representable edge. The product latent has the
range because only its *variance* coordinate is logarithmic while its mean
coordinate is unbounded, which is precisely the combination the wrapped normal
cannot offer.

**Unchanged, because it is structural.** The bridge finding: HELM's time
coordinate at the head is constant (std exactly 0), so dropping it before CALM's
LayerNorm costs nothing. That has nothing to do with the latent.

**Still open.** Whether curvature in the latent helps at all. Nothing here
measures that; it re-specifies what should be measured.

## 5. If the direction is picked up

In order, cheapest first:

1. **Product latent with per-factor learnable curvature** (`LearnablePGMNormal`'s
   shape), Euclidean recovered at `c → 0`. This subsumes the
   hyperbolic-versus-Euclidean question instead of asking it, and connects to
   MiCE, which is HELM's actual contribution.
2. Closed-form KL throughout — no Monte Carlo estimator anywhere in the ELBO.
3. If a single-manifold latent is wanted after all, use the **Poincaré ball**,
   not the Lorentz hyperboloid.
4. Do **not** revive the wrapped normal on the hyperboloid. We have both the
   measurement and the literature saying it is the fragile choice.

The honest prior, though: the Euclidean latent is at 89.11% and is what CALM
ships. Any curved latent has to beat that, and this document is a specification
for a fair attempt, not a prediction that it wins.
