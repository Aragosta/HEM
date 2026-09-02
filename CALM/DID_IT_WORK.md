# Did it work?

Short answer: **the implementation works and is faithful to CALM. The model is
not yet competitive — and the gap is specific to the HELM backbone, not to
CALM's objective.** This corrects a conclusion drawn earlier in `RESULTS.md`.

---

## 1. Is the implementation faithful to CALM?

Checked against `upstream/models/modeling_energy.py` and `train/train_energy.sh`,
not from memory.

| | CALM | this implementation |
| --- | --- | --- |
| alignment | `labels[:, patch_size:]` vs inputs `[:, :-1]` patches | same: patch *p* predicts patch *p+1* |
| target | frozen AE encoder → `(mean, log_std)` | same |
| head | MLP on hidden + uniform noise, zero-init final layer | transcribed |
| energy score | `d_x − 2·d_y`, `beta=1.0`, `n_y=100` | **bit-identical**, gradients included |
| samples | `num_samples=8` | 8 |

`labels = input_ids.copy()` in their trainer, with the shift done inside the
model — which is what my `_aligned` reproduces. The energy score is verified
bit-identical (`experiments/verify_energy.py`).

## 2. But the scale is not comparable, and that matters

| | CALM (published) | these experiments |
| --- | --- | --- |
| sequence length | 8192 | 24–32 |
| patches per sequence at K=4 | 2048 | 6–8 |
| hidden size | 1024 | 33 |
| latent size | 128 | 32 |
| training steps | 250,000 | 4,000 |
| autoencoder | 30,000 steps, ~15B tokens | 800 steps, ~290k tokens |
| vocabulary | 128,256 | 64–97 |

Three orders of magnitude smaller on nearly every axis. These runs can establish
that the mechanism *runs correctly*; they cannot establish that HELM-CALM is a
good model, and the K=4 numbers in particular — six patches of context against
CALM's 2048 — are not measuring the same thing CALM measures.

## 3. What actually happened

Two tasks, same code, same budget.

**The easy task** (arithmetic walk, `t+1 = t + stride mod V`):

| | accuracy |
| --- | --- |
| discrete HELM (cross-entropy) | 99.73% |
| CALM + Euclidean backbone | 99.23% |
| CALM + HELM backbone | **99.09%** |

**The harder task** (tree-structured language, 64 tokens, K=1):

| | accuracy |
| --- | --- |
| discrete HELM (cross-entropy) | 98.79% |
| CALM + Euclidean backbone, width 36 | 98.99% |
| CALM + Euclidean backbone, width 33 (matched to HELM) | **98.89%** |
| CALM + HELM backbone | **90.73%** (85.74% in a separate 2-seed run) |

Read together:

* **CALM's objective is not the problem.** A Euclidean backbone under the energy
  score matches ordinary cross-entropy on *both* tasks. The head, the
  autoencoder, the energy score and the sampling-based evaluation all work.
* **The HELM backbone is where the loss appears**, and only once the task gets
  harder: level on the arithmetic walk, 8–13 points behind on the tree language.

### This corrects an earlier claim

`RESULTS.md` concluded from Stage 1 that "there is no geometric pathology" and
that a hyperbolic backbone trains "as well and as stably as a Euclidean one".
That was true *of the arithmetic walk*, and it does not generalise. On a task
with structure, the hyperbolic backbone falls behind a width-matched Euclidean
one under the same objective.

### One confound checked and ruled out

The Euclidean control was originally built at width 36 while HELM runs at
dim 33 — of which one coordinate is the Lorentz time component, so 32 are usable.
That handed the control a ~12% capacity advantage. Re-running at width 33 gives
98.89%, essentially unchanged. **The gap is not a width artefact.**

### What is not established

One to two seeds per cell, and run-to-run variation on the HELM-CALM cell is
about 5 points (85.74% vs 90.73%). The gap of 8–13 points exceeds that, so it
looks real, but it is not nailed down. Causes untested: the learning rate was
tuned on the arithmetic walk and may not transfer; the head consumes the Lorentz
vector directly rather than via `logmap0` (`NEXT.md` §4.1); the manifold
constraint may genuinely limit what the hidden state can carry into a Euclidean
head.

## 3b. The gap was largely an interface bug — and HELM's own layers fix it

The obvious follow-up: *is a layer missing at the seam between HELM and CALM?*
Yes, and it is a concrete one.

CALM's head opens with `nn.LayerNorm` over all `dim` coordinates. HELM's hidden
state is a Lorentz vector `[x_0, x_1..d]` whose **time coordinate is structurally
`>= sqrt(c)` and never negative** — measured mean 2.46, against a space-part mean
of 0.00. LayerNorm subtracts a per-row mean dominated by a coordinate that is not
a feature, and the result leaves the manifold outright: `<x,x>_L` drifts from
−1.0 to −1.99. Every subsequent `nn.Linear`, `SiLU` and additive residual in that
head assumes a flat space it is no longer in.

Four ways of handing the backbone's output to the head, same task, same budget,
two seeds:

| head input | seed 0 | seed 1 | mean | gap to Euclidean control |
| --- | --- | --- | --- | --- |
| `direct` — the Lorentz vector, as originally built | 90.73% | 84.88% | 87.80% | −11.1 |
| `logmap0` — tangent space at the origin | 91.13% | 91.13% | 91.13% | −7.8 |
| `space` — drop the time coordinate | 95.26% | 94.15% | 94.71% | −4.2 |
| **Lorentz head — built from HELM's own layers** | 93.65% | 96.98% | **95.31%** | **−3.6** |

(References: discrete HELM 98.79%, CALM + Euclidean backbone 98.89%.)

`LorentzEnergyHead` keeps CALM's exact topology — noise embedding, hidden
embedding, four gated residual blocks, final projection — but builds it from
`LorentzLinear`, `LorentzRMSNorm` and `LResNet`, so every intermediate activation
stays on the manifold. Only the last projection leaves it, because the
autoencoder's latent is Euclidean.

**This recovers about 7.5 of the 11.1 point deficit.** Two seeds could not say
whether the remaining ~3.6 points were real, so five seeds were run — see §5.
They are real.

Worth noting that the *principled* bridge (`logmap0`) underperforms the naive one
(`space`). `logmap0` applies a radial warp by hyperbolic distance from the origin;
near the origin the raw space coordinates are already a serviceable Euclidean
chart, and the warp appears to cost more than it buys at this scale. Measured,
not predicted — the reverse was expected.

`head_kind="lorentz"` is now the default.

## 4. So: did it work?

| question | answer |
| --- | --- |
| Is the implementation faithful to CALM? | **Yes** — verified against their code, energy score bit-identical |
| Does it run end to end? | **Yes** — 15 integration tests, gradients reach every component |
| Does a hyperbolic backbone train under the energy score? | **Yes** |
| Does it match a Euclidean backbone? | **No, but much closer** — was 11.1 points behind, now 2.94 at 4.6 standard errors (§5) |
| Was a layer missing at the HELM/CALM seam? | **Yes** — CALM's Euclidean LayerNorm on a manifold point. A head built from HELM's own hyperbolic layers recovers ~7.5 of the 11.1 points |
| Is HELM-CALM a good language model? | **Unknown, and untestable at this scale** |
| Does patching preserve HELM's hierarchy? | **Unknown** — the instrument failed, see `HIERARCHY.md` |

The finding that matters for `NEXT.md`: what looked like "the hyperbolic backbone
underperforms under CALM's objective" was mostly **an interface bug, not a
property of the geometry**. Feeding a manifold point into a Euclidean LayerNorm
cost ~7.5 points; building the head from HELM's own layers recovers them. A
2.94-point remainder survives, and five seeds show it is not noise (§5).

That is a much better position than the previous section implied, and it
generalises a warning to the rest of the design: **anywhere CALM's Euclidean
machinery touches HELM's manifold activations, check the geometry before
trusting the number.** The patch embedding was already built this way; the head
was not, and that was worth 7.5 points.


## 5. Five seeds: the remainder is real

Two seeds left the residual gap ambiguous. Five settle it. Tree language, K=1,
4000 steps, identical data and schedule, only the seed varying:

> **Correction (see `EVALUATION.md` §1).** The script behind this table trains
> and evaluates on the *same* sixteen batches — 1024 tokens seen ~250 times. The
> numbers below are **training-set accuracies**, so they measure memorisation
> capacity under the energy objective, not generalisation. Both arms are measured
> identically, so the 2.94-point *difference* stands; the absolute figures do not
> mean what an accuracy on held-out text would mean, and nothing here establishes
> which architecture is the better language model.

| | s0 | s1 | s2 | s3 | s4 | mean | sd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CALM + HELM (Lorentz head) | 93.65% | 96.98% | 96.47% | 96.67% | 96.98% | **96.15%** | 1.41% |
| CALM + Euclidean (control) | 98.89% | 99.09% | 99.19% | 99.19% | 99.09% | **99.09%** | 0.12% |

**Difference 2.94 points, standard error 0.63, ratio 4.6.** That is not seed
noise. The earlier "within seed noise" reading came from having only the two
extreme HELM seeds and no control spread to compare against.

The second number is arguably the more interesting one: the Euclidean control's
seed spread is **0.12%**; HELM-CALM's is **1.41%**, roughly twelve times larger.
Whatever is costing the 2.94 points is also making the model markedly less
stable across initialisations. A systematic 3-point handicap and a 12× variance
inflation are more consistent with one shared cause than with two.

### Where the geometry still breaks

After the head fix, trace what is on the manifold and what is not:

| component | geometry |
| --- | --- |
| token embedding | hyperbolic (`ManifoldParameter` on the hyperboloid) |
| patch embedding | hyperbolic (`LorentzPatchEmbedding`, space-concat + time recompute) |
| backbone (HMLA + MiCE) | hyperbolic throughout |
| energy head | hyperbolic (`LorentzEnergyHead`) — **until its last layer** |
| **autoencoder latent** | **Euclidean** |

There is exactly one seam left, and it is forced. `LorentzEnergyHead.forward`
ends with

```python
return self.final(..., return_space=True)
```

It drops off the manifold at the final projection not because that is the right
modelling choice, but because the target it must match — the frozen
autoencoder's `(mean, log_std)` latent — lives in Euclidean space. The energy
score `E‖X−y‖^β − ½E‖X−X′‖^β` is then computed with a Euclidean norm on a
quantity the rest of the model produced hyperbolically.

So the remaining gap sits precisely where the geometry is still mixed. That is a
hypothesis, not a demonstration — but it is now a hypothesis with a **measured
2.94-point target** rather than a hypothetical one, which is what a hyperbolic
latent (GM-VAE-style, where `(μ, log σ²)` under the Fisher metric *is* a point in
H², or a Lorentz VAE) would have to beat to justify itself.

`gmvae/ASSESSMENT.md` deferred that work for the wrong reason — instability that
later turned out to be a learning-rate problem. The right reason to pick it up
now is this table.
