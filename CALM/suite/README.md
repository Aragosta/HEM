# The HELM × CALM test suite

A design, then an implementation. The design is the part that matters: this
project has repeatedly produced numbers that could not answer the question they
were run for, and almost every one of those failures was decided before any code
ran.

## 1. What each system claims

**CALM** ([2510.27688](https://arxiv.org/abs/2510.27688)) claims a *better
performance–compute tradeoff* by predicting one continuous vector per K tokens
instead of one distribution per token. Two halves:

* **efficiency** — K-fold fewer autoregressive steps. Structural: true by
  construction, already verified here (K=4 gives 4× fewer steps, attention at
  0.062×). It is also true of a model that emits garbage, so it is not evidence
  of anything on its own.
* **quality retained** — the hard half, resting on two things: a *near-lossless*
  autoencoder, and an energy-score objective that is strictly proper so the head
  learns the true conditional distribution without a likelihood.

**HELM** ([2505.24722](https://arxiv.org/abs/2505.24722)) claims that hyperbolic
geometry matches the hierarchical, power-law structure of language, and shows it
as a *quality gain at matched parameters* against a Euclidean (DeepSeek-style)
baseline on standard LM benchmarks.

**So HELM-CALM's implicit claim is a conjunction**: CALM's step reduction *and*
HELM's quality advantage. Neither paper tests it, because the interesting
possibility is that they interact — patching K tokens into one vector may
destroy precisely the token-level hierarchy that makes HELM better.

## 2. The design: a 2×2, because the question is an interaction

Every previous experiment here compared two things and could not attribute the
difference. The fix is a factorial:

|                        | discrete head (cross-entropy) | CALM head (energy score) |
| ---------------------- | ----------------------------- | ------------------------ |
| **hyperbolic backbone** | `helm_discrete` — HELM as published | `helm_calm` — the model under test |
| **Euclidean backbone**  | `euclid_discrete` — the baseline HELM's paper beats | `euclid_calm` — CALM as published |

Four cells, and three separable quantities:

* **geometry main effect** = (helm_\* − euclid_\*) — does hyperbolic help at all,
  in our hands? If this is ~0 in the *discrete* column, we have failed to
  reproduce HELM, and nothing in the CALM column can be interpreted.
* **objective main effect** = (\*_calm − \*_discrete) — what CALM's objective
  costs or buys, independent of geometry.
* **interaction** = (helm_calm − euclid_calm) − (helm_discrete − euclid_discrete)
  — **the actual question.** Does hyperbolic geometry help *more*, *less*, or the
  same once you predict patches instead of tokens? A negative interaction is the
  concrete form of "patching destroys the hierarchy HELM needs".

`helm_discrete` is the load-bearing cell. It is our reproduction check against
HELM's own paper: if the geometry effect is absent there, the suite reports that
and stops rather than proceeding to a comparison that cannot mean anything.

**Matching.** Cells must differ in one thing only. The Euclidean backbone's width
is searched to match HELM's **trainable parameter count** within 2%, and both
**active parameters per token** (HELM's MoE routes only some experts) and a FLOP
estimate are reported alongside, since a parameter match and a compute match are
not the same and the reader needs both. Heads, latent, autoencoder, optimizer,
schedule, data order and seeds are shared exactly.

## 3. Metrics, and which comparisons each licenses

The trap this suite is built to avoid: reporting a metric in a cell where it does
not exist, or comparing two metrics that are not on the same scale.

| metric | cells | comparable across | what it is for |
| --- | --- | --- | --- |
| **bits per byte** | discrete only | the discrete column | the standard LM number. **Structurally impossible for CALM** — an implicit sampler has no density. Not an omission. |
| **top-1 next-byte accuracy** | all four | **all four** | the one quality number on the same scale everywhere. Carries the 2×2. |
| **BrierLM, per order** | all four | all four | the proper score that survives having no likelihood. Reported per n-gram order, never only as the geometric mean, which reads 0.0000 for every model alike at byte level. |
| **bigram / unigram / uniform baselines** | — | all four | a model that loses to a lookup table has not learned the corpus. Non-negotiable after a previous run scored 9.23% held-out against a bigram table's 19.86%. |

**Training-dynamics metrics** (the "is it learning, and how"): steps to reach a
loss threshold, gradient-norm percentiles, and the **effective rank** of hidden
activations (participation ratio of the covariance spectrum). Effective rank is
the cheap detector of representational collapse — a model can have a falling loss
and a rank-3 representation.

**Efficiency**: autoregressive steps per token, wall-clock per token, peak
memory, and a FLOP estimate. Reported per cell so the tradeoff CALM claims is
measured rather than assumed.

## 4. The "why" probes

Accuracy tells you *what*. These are for *why*, and each is tied to a specific
mechanism that has already bitten this project:

* **`radius`** — how far activations sit from the origin, per layer. Float32 on
  the hyperboloid fails past radius ≈ 8 (measured: constraint error 0.25 at
  radius 8, 1.0 at radius 10). HELM's own activations sit at 2.8–3.1. Any
  hyperbolic number measured at radius > 6 is suspect *before* it is interpreted.
* **`clamped`** — fraction of activations pinned against any clamp. Near 1 means
  the accuracy is a measurement of a hyperparameter.
* **`delta_hyperbolicity`** — Gromov's four-point condition on the learned
  representations, diameter-normalised. Lower is more tree-like. This is HELM's
  thesis stated as a measurable property: if the hyperbolic backbone is not
  producing more tree-like representations than the Euclidean one, the geometry
  is not doing what it is supposed to, whatever the accuracy says.
* **`patch_delta`** — the same quantity measured on **patch** representations
  rather than token representations. This is the direct test of the hierarchy
  worry: if δ rises sharply from token level to patch level, patching is
  flattening the structure HELM depends on.
* **`kl` / `collapse`** — latent usage and mode collapse. A negative BrierLM
  means the two independent draws agree with each other but not the target;
  that is collapse, not "worse", and it is reported as such.
* **`nonfinite`** — any NaN or Inf step invalidates its row outright. Three
  results in this project were numerical failures wearing the costume of an
  architectural finding.

## 5. Tiers, so the GPU is spent only where it is needed

Each tier answers a different question, and a failure at a lower tier makes the
higher one pointless.

| tier | hardware | scale | question it answers |
| --- | --- | --- | --- |
| **0 — smoke** | CPU, minutes | tiny, ~30 steps | does every cell run, is every metric finite and non-vacuous |
| **1 — CPU study** | CPU, hours | ~1M params, 900KB bytes, K=4 | the full 2×2 with seeds. Establishes signs and the noise floor. **Not** a claim about language models at scale. |
| **2 — single GPU** | 1 GPU, hours | 120M preset, real tokenizer, WikiText-scale | does the tier-1 sign survive at a scale where architecture comparisons start to be predictive |
| **3 — comparable** | multi-GPU | HELM's published setting | the only tier whose numbers are comparable to the paper tables |

Tier 1 is designed to be run *now*, on 4 CPU cores, and to be honest about being
a small-scale study rather than dressed up as more. Tier 2 is the first tier
worth GPU time, and the suite emits its config so it can be launched unchanged.

## 6. What would falsify what

Stated in advance, so results cannot be reinterpreted after the fact:

* **geometry effect absent in the discrete column** → we have not reproduced
  HELM; the CALM column is uninterpretable, and the suite says so rather than
  reporting the CALM comparison.
* **interaction ≈ 0** → patching neither helps nor harms the hyperbolic
  advantage; HELM-CALM is just CALM with a hyperbolic backbone, and the
  efficiency case stands or falls on its own.
* **interaction < 0** → patching costs the hyperbolic advantage. The concrete
  version of the `HIERARCHY.md` worry, and it should show up as `patch_delta`
  rising above the token-level δ.
* **interaction > 0** → hyperbolic geometry helps *more* under CALM's objective.
  Would be the strongest possible case for HELM-CALM, and the one demanding the
  most scepticism: check `radius`, `clamped` and `nonfinite` before believing it.
* **any cell below the bigram baseline** → that cell has not learned the corpus,
  and its row is descriptive of nothing.
