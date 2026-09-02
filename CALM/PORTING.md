# How CALM should be ported to HELM — reading their code, not our assumptions

Written after going back to `upstream/models/` and to the hyperbolic-network
literature, prompted by the observation that we were overlooking something. We
were. Three of the four things below invalidate work already done in this repo.

## 1. HELM already leaves the manifold at its output. Our "seam" was its design.

`DID_IT_WORK.md` §5 identified the last Euclidean seam as
`LorentzEnergyHead.forward` ending in `return_space=True`, and called it *forced*
— an unwanted compromise imposed by the Euclidean latent. That framing was wrong.
Here is HELM's own vocabulary head, unmodified, in `helm/modules/helm_mice.py`:

```python
h = self.norm(self.final_proj(h, return_space=True), space_only=True)
...
return self.head(h)          # a plain Euclidean nn.Linear
```

**HELM drops the time coordinate and applies a Euclidean linear head.** That is
not a concession; it is how the architecture is designed to emit predictions.
Our energy head doing the same thing at its last layer is *faithful to HELM*, not
a defect in need of repair. The 2.94-point gap does not live there, and the
hyperbolic-latent work was aimed at a target that was never a problem.

## 2. CALM's latent is already a tangent space

CALM's autoencoder is a VAE with a KL pull toward `N(0, I)`
(`modeling_autoencoder.py`):

```python
kl_loss = 0.5 * (mean.pow(2) + std.pow(2) - 1 - log_std * 2)
kl_loss = torch.clamp(kl_loss, min=self.kl_clamp)
```

So the latent is bounded, centred and approximately isotropic by construction.
That is *exactly* the structure of the tangent space at the origin,
`T_o H^n ≅ R^n` — which is **genuinely Euclidean**, not an approximation to
something curved. Distances in it are ordinary L2 distances, and the energy
score's `||x − y||^β` is already the correct Riemannian quantity there.

There was never a geometric mismatch at the latent. We invented one, built a
wrapped-normal VAE to fix it, and introduced a float32 failure in the process.

## 3. The literature says to do it the way we did not

The standard construction for hyperbolic VAEs is to *"perform variational
inference in an intermediate Euclidean space and then map the result to the
hyperbolic disk through a fixed, deterministic transformation, with the KL
divergence computed in the Euclidean latent space between variational Gaussians
and standard normal priors, which has a closed-form solution ensuring stable
optimization"* — [Autoencoding Hyperbolic Representation for Adversarial
Generation](https://arxiv.org/pdf/2201.12825).

That is the opposite of what `hyperbolic_latent.py` does. The wrapped normal has
**no closed-form KL** (we estimate it by Monte Carlo) and **no bound on radius**
(we had to add a clamp after it walked to radius 9.46 and produced NaN). The
literature's design avoids both problems by construction. Our instability was
not bad luck; it was the predictable cost of the fragile option.

Note also, from [Fully Hyperbolic Neural
Networks](https://arxiv.org/abs/2105.14686), that mapping to the tangent space is
considered a *limitation* when done for the network's internal operations — but
the exception it names is precisely *"when we need to map the input from the
Euclidean space or the output to the Euclidean space."* Input and output are
where flat maps are legitimate. That is exactly where CALM touches HELM.

## 4. Concrete gaps between our port and CALM's actual code

Read side by side, our implementation is thinner than theirs in ways that matter
and that have nothing to do with geometry:

| piece | CALM (`upstream/`) | ours | why it matters |
| --- | --- | --- | --- |
| patch compression | `Linear(K·H, 2H)` → `SiLU` → `Linear(2H, H)` → `LayerNorm` | **one** `LorentzLinear` | this is where K tokens become one vector — exactly where the hierarchy question lives, and we gave it a single linear map |
| head input | separate `norm_noise` and `norm_hidden` LayerNorms on the two branches | no entry normalisation of either branch | the noise and the condition enter at different scales |
| autoencoder KL | `kl_clamp = 0.5` (free bits, per dimension) | none | without free bits the KL collapses unused dimensions |
| autoencoder latent | `ae_dropout = 0.15` on the sampled latent | none | forces the decoder to tolerate the noise the head will produce |
| latent width | `latent_size = 128` | 32 in our experiments | four times narrower than the design point |

The patch-compression row is the important one. CALM spends a two-layer gated MLP
turning K token vectors into one; we spend a single linear. `HIERARCHY.md` asks
whether patching destroys the hierarchy HELM exists to model — and we tried to
answer it with a patch embedding a quarter the depth of the one CALM found
necessary.

## 5. What the port should actually be

Keep the geometry where it earns its place and stop forcing it where it does not:

1. **Backbone stays fully hyperbolic.** HMLA and MiCE on the manifold, unchanged.
   This is where token hierarchy lives and where HELM's thesis applies.
2. **Patch embedding stays hyperbolic but gets CALM's depth** — a two-layer gated
   map built from `LorentzLinear`, not one projection. If patching is where the
   hierarchy is lost, this is the layer that decides it.
3. **Bridge at the output via `logmap0`**, and treat the result as what it is: a
   tangent vector at the origin, a genuinely Euclidean object.
4. **Head and latent stay CALM's, unmodified.** Euclidean MLP, Euclidean latent,
   Euclidean energy score. Not a compromise — the tangent space is flat, and this
   is the case the literature names as the legitimate one.
5. **Retire the wrapped-normal latent** as the primary direction. Keep the module:
   the geometry in it is verified and the float32 radius finding is worth having.
   But it solves a problem that does not exist and costs stability to do it.

One measurement contradicts step 3 and must be redone before it is trusted:
`logmap0` previously scored *worse* than naive space-dropping (91.13% vs 94.71%).
That was measured on the tree language, whose control we now know fails to
generalise, with a head we have since replaced. It is not evidence about
anything, in either direction.

## 6. What this costs us

Honestly accounted:

- `hyperbolic_latent.py` — the wrapped normal, the geodesic energy score, the
  propriety proof — is **correct code aimed at a non-problem**. It stays, tested,
  as a negative result and for the radius finding.
- The `DID_IT_WORK.md` §5 conclusion that "there is exactly one seam left" is
  **wrong**: that seam is HELM's own output design.
- The "hyperbolic geometry vs CALM's objective" question is still open, but the
  experiment that should answer it needs the patch embedding of row 2 above, not
  a different latent.
