# Can CALM help HELM-MiCE?

**Short answer: the fit is unusually good on cost, and there is one compatibility
accident that removes the biggest prerequisite — but it breaks HELM's evaluation
protocol outright, and one component has an unresolved theoretical question.
Worth prototyping; not a drop-in.**

---

## 1. Why the fit is good

CALM removes the softmax-over-vocabulary head and replaces next-token prediction
with next-*patch* prediction of a continuous latent. HELM-MiCE happens to be
almost pathologically exposed to exactly that cost.

From our own profiling (`docs/UPGRADES.md`): HELM's `dim=390` meets a
128256-entry Llama-3 vocabulary, so `head` is **50M of a 107M model** and
**81% of the forward pass**, and it materialises **3.9 GiB** of float32 logits at
the released training shape.

Measured at HELM's shape (`python CALM/estimate_helm_calm.py`):

| | HELM vocab head | CALM head |
| --- | --- | --- |
| parameters | 50.02 M | **4.04 M** (12× smaller) |
| output activation (2×1024 tok) | 1002 MiB | **8 MiB** (125× smaller) |
| forward + backward | 5142 ms | **1698 ms** (3.03×) |

The CALM figure already includes the 8 energy samples per position its training
objective requires, so it is a fair comparison rather than a best case.

On top of that, the backbone sees `seq/K` positions:

| K | positions (from 1024) | attention scores | everything else | AR steps / 2048 tok |
| --- | --- | --- | --- | --- |
| 2 | 512 | 0.25× | 0.50× | 1024 |
| 4 | 256 | **0.06×** | 0.25× | **512** |
| 8 | 128 | 0.02× | 0.125× | 256 |

The paper reports **>40% lower FLOPs** than a standard transformer at matched
performance. For HELM the head saving alone is larger than that, because HELM's
head is a bigger share of its model than a normal LLM's is.

Three secondary effects, all favourable:

* **The KV cache shrinks by another factor of K** on top of the 6.1×/14.1× the
  latent MLA cache already gives, since there are K× fewer positions to cache.
* **MoE routing runs K× less often** — one routing decision per patch instead of
  per token.
* **Our fused cross-entropy head becomes unnecessary.** The 2.09× training-step
  win it produced is subsumed: there is no vocabulary projection left to fuse.

## 2. The compatibility accident that matters

**CALM's released autoencoder is tokenizer-compatible with HELM out of the box.**

Both use the Llama-3.1-8B tokenizer. CALM's vendored `llama3_tokenizer` has
128000 + 256 = **128256** entries; HELM's `vocab_size` is **128256** and its
training script loads `meta-llama/Llama-3.1-8B`. Same tokenizer, same vocabulary,
same indices.

This removes the single largest prerequisite. CALM's pipeline normally requires
pretraining a 75M-parameter autoencoder on ~15B tokens *before* the language
model can be trained at all. Because the tokenizers match, the
[released CALM autoencoder](https://huggingface.co/cccczshao/CALM-Autoencoder)
can be reused directly — the latent space it defines is over exactly the token
ids HELM already emits. A prototype needs no autoencoder training run.

(Worth verifying before relying on it: the checkpoint's config should report
`vocab_size=128256` and `patch_size=4`. Zenodo and Hugging Face are both blocked
from this machine, so this was established from the tokenizer files in the repo,
not from the checkpoint itself.)

## 3. What breaks

### 3.1 HELM's entire evaluation protocol

This is the most concrete problem and it is not a detail.

CALM is **likelihood-free**. It cannot score a continuation's log-probability,
which is precisely what `lm-evaluation-harness` multiple-choice tasks require:
`loglikelihood` compares candidate answers by their log-probability. HELM's
paper — and our `helm/eval/` integration — evaluates on MMLU, ARC-Challenge,
HellaSwag, CommonsenseQA and OpenBookQA through exactly that call.

Under CALM, `loglikelihood` cannot be implemented as such. CALM substitutes
**BrierLM**, estimated from black-box samples. The consequences:

* Our `score_continuations` path does not carry over; a CALM-HELM would need
  the sampling-based estimator from `upstream/models/modeling_calm.py::eval_brier`.
* **The paper's Table 1 comparison against DeepSeek-V3 could not be reproduced
  on a CALM-HELM.** BrierLM and multiple-choice accuracy are different metrics;
  there is no conversion. Any claim that CALM "helped" would need a new baseline
  measured the same way — i.e. a discrete HELM scored by BrierLM too.

Budget for that: it is a second evaluation stack, not a flag.

### 3.2 A hyperbolic backbone meeting a Euclidean latent

HELM's hidden states are Lorentz vectors `[x_t, x_s]` with `x_t = √(|x_s|² + c)`.
CALM's head consumes a Euclidean hidden state and predicts a Euclidean latent.
Two ways to bridge:

* **Euclidean latent (pragmatic).** Map HELM's final hidden state to the tangent
  space at the origin (`logmap0`) and feed CALM's head unchanged. The
  autoencoder stays Euclidean and reusable. This is the version to prototype:
  it isolates one variable, and the head is the part with measured benefit.
* **Hyperbolic latent (research).** Predict a point on the manifold and use a
  Lorentzian distance in the energy score. More faithful to HELM's thesis, and
  the source of the open question below.

### 3.3 The open theoretical question: is the energy score still proper?

CALM's objective is the energy score

```
S(P, y) = E‖X − y‖^β − ½·E‖X − X′‖^β,      X, X′ ~ P
```

which is a *strictly proper* scoring rule for `β ∈ (0, 2)` — the property the
whole likelihood-free framework rests on. That result depends on `‖·‖^β` being a
negative-definite kernel on Euclidean space (Schoenberg).

Substituting a hyperbolic distance does **not** automatically preserve strict
propriety. Real hyperbolic space is known to be a metric space of negative type,
which is suggestive, but HELM does not use the geodesic distance — it uses the
*squared Lorentzian* distance, a different object, and the relevant β-range would
need to be established rather than assumed.

**This is checkable, and should be checked before building anything on it.** It
only blocks the hyperbolic-latent variant; the Euclidean-latent version in 3.2
sidesteps it entirely by keeping the energy score in the space it was proved for.

### 3.4 A conceptual tension worth stating

HELM's thesis is that hyperbolic geometry matches the **token-level** hierarchy
of language. CALM's thesis is that K tokens can be compressed into one vector
with negligible loss.

These are not obviously compatible. Compressing four tokens into a latent may
blur exactly the token-level hierarchical structure HELM exploits — or the latent
space may turn out to be *more* hierarchical than token space, since it encodes
phrases rather than word pieces, which would favour HELM more strongly. Nobody
knows; it is the interesting question here, and it is empirical.

### 3.5 Unknowns that arithmetic cannot settle

* No hyperbolic model has been trained under an energy-score objective. The
  interaction between the manifold constraint (every activation renormalised onto
  the hyperboloid) and a likelihood-free sampling objective is unexplored.
* HELM's numerical fragility is real — this port fixed clamping and offset bugs
  in the released code. An objective built on sampled distances may be less
  forgiving than cross-entropy.
* MoE load balancing was designed per token. With K× fewer, coarser routing
  decisions, the balancing hyperparameters would need re-tuning.

## 4. A staged plan

Ordered so that each stage produces a decision, and the cheap checks come first.

**Stage 0 — verify the free lunch (hours).** Download the CALM autoencoder,
confirm `vocab_size=128256`, and check reconstruction accuracy on HELM's
Wikipedia training data with HELM's tokenizer. CALM reports >99.9% on the Pile.
If it does not transfer to this corpus, everything downstream needs an
autoencoder training run and the cost picture changes.

**Stage 1 — swap the head only, K=1 (days).** Keep HELM exactly as it is, replace
the vocab head with CALM's MLP generator and energy loss at patch size 1. This
tests the objective in isolation, with no compression and no sequence-length
change. If a hyperbolic backbone will not train under an energy score, it fails
here, cheaply.

**Stage 2 — K=4 with the Euclidean latent (weeks).** Add patching: `logmap0` from
HELM's manifold into the tangent space, CALM's head unchanged, frozen CALM
autoencoder. Compare against discrete HELM **on BrierLM**, having first measured
the discrete baseline on BrierLM too — per 3.1 there is no shortcut.

**Stage 3 — hyperbolic latent (research).** Only if 3.3 resolves favourably, and
only if Stage 2 shows the geometry is doing work.

## 5. Recommendation

**Prototype Stages 0–1.** The measured head numbers (12× fewer parameters, 125×
smaller activation, 3.03× faster) are large enough, and HELM's head is a large
enough fraction of the model, that this is one of the highest-leverage changes
available to it — larger than anything remaining on the optimization side, where
the next step is a Triton kernel worth maybe 20%.

But go in with the cost understood: **this is a research project, not an
optimization.** It changes what the model predicts, requires a second evaluation
stack, and severs comparability with the HELM paper's headline table. The
optimization work in `docs/OPTIMIZATIONS.md` and `docs/UPGRADES.md` is
provably output-preserving; this is not, and cannot be.
