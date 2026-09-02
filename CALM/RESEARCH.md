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

So the correct reading of our 2.29% patch reconstruction is not "hyperbolic
latents do not work". It is "we picked the parameterisation the literature warns
about, and got the failure the literature predicts."

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

**Unchanged, because it is measured.** With a *working* wrapped-normal
implementation (clamped, finite, no NaN), the Euclidean autoencoder reconstructs
held-out K=4 byte patches at **89.11%** against the hyperbolic one's **2.29%**.
That number stands as a fact about our implementation. It is now attributable to
the parameterisation rather than to the geometry.

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
