# GM-VAE for CALM-HELM

**Short answer: the idea fits remarkably well and is worth pursuing, but the
specific mechanism GM-VAE offers solves the *density* problem, not the *energy
score* problem — and the one piece of code HELM would most want is broken in the
release.**

---

## 1. The fit is better than it looks

CALM's autoencoder emits `(mean, log_std)` per latent dimension, and its energy
score samples targets as `mean + eps * std`. That is a Euclidean Gaussian in
R^latent.

GM-VAE's observation is that a `(μ, log σ²)` **pair**, under the Fisher
information metric, *is* a point in 2-dimensional hyperbolic space. So CALM's
`latent_size=128` latent, read as 64 such pairs, is already a point in `(H²)^64`.

**A hyperbolic latent therefore requires no change to CALM's autoencoder output
shape** — same tensor, same head, different distance. That is an unusually cheap
interface between two papers that have nothing to do with each other, and it is
what makes this worth testing at all.

It is also the natural pairing for HELM specifically: HELM's backbone is already
hyperbolic, so a hyperbolic latent removes the `logmap0`-to-Euclidean bridge that
Stage 2 would otherwise need (`../ASSESSMENT.md` §3.2).

## 2. But it does not answer the energy-score question

`../ASSESSMENT.md` §3.3 flagged the open question: the energy score

```
S(P, y) = E‖X − y‖^β − ½·E‖X − X′‖^β
```

is strictly proper for `β ∈ (0, 2)` because `‖·‖^β` is a negative-definite kernel
on Euclidean space. GM-VAE does **not** resolve this, and it is important not to
read it as if it does:

* GM-VAE's stability comes from writing its log-density with a **Gaussian KL
  divergence**, a local approximation of the squared Fisher–Rao distance.
* **KL is not a metric.** It is asymmetric and violates the triangle inequality.
  Substituting it for `‖·‖` in the energy score breaks propriety outright — the
  rule would no longer be minimised at the true distribution.

So GM-VAE gives a numerically stable *density* on a hyperbolic latent space,
which is exactly what a VAE needs, and not what a likelihood-free scoring rule
needs. Using it would mean either:

* keeping the energy score and using the true **Fisher–Rao distance** (the
  hyperbolic distance on H²), which puts us back on the original open question —
  though now with GM-VAE's coordinates, where that distance is at least
  well-conditioned; or
* dropping the energy score for a likelihood-based objective on the GM-VAE
  latent, which abandons CALM's likelihood-free framework and its BrierLM
  evaluation along with it.

Neither is a small decision, and the second one undoes much of why CALM was
attractive.

## 3. What is broken in the release

The single most useful piece of GM-VAE for HELM is its **Lorentz ↔ half-plane
conversion** — HELM's activations are Lorentz vectors, GM-VAE's latents are
half-plane `(μ, log σ²)` coordinates, and converting between them is precisely
the interface a hyperbolic CALM-HELM needs.

`upstream/distributions/PGMNormal/layers.py` contains exactly that, in
`GeoEncoderLayer` and `GeoDecoderLayer`:

```python
mean = self.manifold.expmap0(F.pad(mean, (1, 0)))
mean = lorentz2halfplane(mean, self.c, log=torch.Tensor([True]))
...
z = halfplane2lorentz(z, self.c)
z = self.manifold.logmap0(z)[..., 1:]
```

**`lorentz2halfplane` and `halfplane2lorentz` are never defined or imported
anywhere in the repository.** A static check of the module confirms both names
are called but unbound, so `--layer Geo` raises `NameError` on the first forward
pass. The sibling `LearnablePGMNormal/layers.py` imports its equivalents from a
`dgnn` package that is not vendored and not in `requirements.txt`.

The paths the paper's own reproduce commands exercise (`--dist=PGMNormal` with
the default `Vanilla` layers) do not touch these, so the release is fine for
reproducing the paper. It is not fine for the thing HELM would want from it.

The conversion is standard textbook geometry and reimplementing it is a few lines
— but it has to be written and tested, not imported, and that changes the cost
estimate for anyone assuming the code is drop-in.

## 4. What GM-VAE *does* offer HELM, concretely

Setting aside the energy score, two things stand up:

1. **A numerically stable hyperbolic parameterization.** GM-VAE's whole point is
   avoiding `expmap`/`logmap`/`arcosh` near their singularities. HELM's Lorentz
   layers are full of defensive clamps (`clamp_min(1e-4)` and friends) for
   exactly that reason, and this port has already fixed real numerical bugs in
   them. A `(μ, log σ²)` representation sidesteps that class of problem entirely.
2. **A hyperbolic prior with a closed-form KL.** If a CALM-HELM ever wants a
   regularised latent — CALM's autoencoder already carries a KL term with weight
   `1e-3` — GM-VAE supplies the hyperbolic version in closed form
   (`upstream/distributions/utils.py`), with no sampling and no special functions
   beyond `lgamma`/`digamma`.

## 5. Recommendation

**Not the next thing to do.** The Stage 1 instability turned out to be a learning
rate, not a geometric pathology (see `../RESULTS.md`), so the motivation that
made GM-VAE look urgent — "hyperbolic latent-variable models are unstable, and
here is the fix" — does not apply to the problem we actually have.

Revisit it at Stage 3, if Stage 2 shows the Euclidean latent is the limiting
factor. At that point GM-VAE is the right starting point for the representation,
with two pieces of work attached that the release does not provide: the
Lorentz ↔ half-plane conversion, and a decision about what replaces the energy
score's Euclidean norm.
