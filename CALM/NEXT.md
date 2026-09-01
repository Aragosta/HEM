# What a proper HELM-CALM needs

Everything so far lives in `experiments/` and runs on a 33-dimensional toy model.
That was the right shape for the questions being asked — *does the objective
train a manifold-constrained backbone, does the patching path work, is the
autoencoder good enough* — and all three are now answered. None of it is a model.

This is what stands between here and one.

---

## 1. What the evidence has already settled

Not open questions any more; these are constraints to build against.

| | settled by |
| --- | --- |
| The energy score trains a hyperbolic backbone, as stably as a Euclidean one | Stage 1 + lr sweep: 99.09% vs 99.23%, spread 1.36 vs 1.22 |
| **lr ≈ 1e-3**, not cross-entropy's 3e-3 | at 3e-3 the seed spread is 24.5 points, at 1e-3 it is 1.4 |
| **β = 1.0**, never below | below 1 the self-distance term has unbounded derivative → NaN gradients, in CALM's code too |
| `ManifoldParameter`s need a Riemannian optimizer | plain AdamW drove `\|⟨x,x⟩_L + c\|` to 3.92 |
| The autoencoder is tokenizer-compatible | 128256 = 128256, every WikiText id inside HELM's embedding |
| Autoencoder quality is a budget question, not a ceiling | 86.29% → 97.55% over a 10x budget sweep, still climbing |
| The K>1 patching path works | 99.18 / 97.80 / 88.28% at K = 1 / 2 / 4 |
| BrierLM is the comparison metric, and works for both model types | validated against `V^-k` closed form |

---

## 2. The code that has to exist

Today the pieces are experiment scripts. A model needs them promoted into the
library, with the repository's usual standard — tests, docstrings, checkpoint
compatibility.

| module | from | what it is |
| --- | --- | --- |
| `helm/modules/patch_embedding.py` | `stage2a_patching.py` | `LorentzPatchEmbedding` — K Lorentz vectors → one, staying on the manifold |
| `helm/modules/energy_head.py` | `stage1_energy_head.py` | `CalmHead` + `energy_score`, verified bit-identical to CALM's |
| `helm/modules/helm_calm.py` | new | the model: patch embed → HELM backbone → energy head, frozen AE alongside |
| `helm/eval/brierlm.py` | `experiments/brierlm.py` | already validated; needs the lm-eval adapter |
| `helm/eval/autoencoder.py` | new | load CALM's released checkpoint, freeze, expose `encode`/`decode` |
| `train_calm.py` | new | the training loop, with the optimizer split below |

**`HelmCALM.forward` has to serve three callers**, and they want different
things: training wants the energy loss, BrierLM evaluation wants two independent
sampled token sequences, and generation wants one patch at a time with a KV
cache. Upstream CALM conflates the first two inside `forward` via an
`if not self.training` branch that returns a different dataclass — the same
pattern that makes the released `LorentzMoE` unusable in eval mode. Keep them as
separate methods.

**The optimizer split is not optional.** Three groups: `ManifoldParameter`s on
RiemannianAdam (HELM's `train.py` already does this and the CALM path must not
regress it), everything else on AdamW at ~1e-3, and the autoencoder frozen with
`requires_grad_(False)` and in `eval()` — CALM sets both, and forgetting `eval()`
leaves dropout/norm statistics live in what is supposed to be a fixed decoder.

---

## 3. The scale finding: 120M is the wrong shape for this

The parameter accounting, computed from the real modules:

| | embed | head | layers | **HELM total** | CALM head | **CALM LM** | + frozen AE | AE share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 120M preset | 50.0M | 50.0M | 12.5M | **112.7M** | 4.04M | **66.7M** | 142.5M | **53.2%** |
| 1B preset | 114.8M | 114.9M | 202.8M | **433.3M** | 21.1M | **339.5M** | 415.3M | **18.3%** |

(The 1B preset is the *evaluation* config; the training script disagrees with it
on three fields — see `helm/eval/presets.py::KNOWN_TRAIN_EVAL_MISMATCHES`.)

Two things follow, and the first is easy to miss:

**CALM removes the head, not the embedding.** `embed_tokens` over the full
128256-entry vocabulary is still needed to encode the input. So the LM shrinks by
the head's 50M, not by both — 112.7M → 66.7M, not to 12.5M.

**At 120M the frozen autoencoder is 53% of the deployed system, and the system is
larger than plain HELM** (142.5M vs 112.7M). The autoencoder is a fixed 75.8M
cost that does not shrink with the model. For comparison, in CALM's own lineup it
is 17.0% of CALM-M, 9.3% of CALM-L, 4.0% of CALM-XL — they operate where it is a
rounding error.

So: **run Stage 2 at the 1B preset.** There the trained LM is 339.5M against
HELM's 433.3M and the whole system is smaller too. At 120M the win is confined to
training compute (no 128k softmax in the backward, K-fold shorter sequences), and
the deployment story is worse than plain HELM. Reporting a 120M HELM-CALM as an
efficiency result would be misleading.

---

## 4. Open decisions

**4.1 How the backbone's output reaches the head.** Stage 1 fed HELM's final
Lorentz vector straight into the head's first `Linear`. That is what HELM's own
vocab head does to the same tensor, so it is consistent with the model's existing
design, and it worked. The alternative is `logmap0` into the tangent space at the
origin, which is the more principled reading of "take a manifold point to a
Euclidean vector". *Recommendation: keep the direct path, run logmap0 as a
one-line ablation.* Cheap, and it settles an objection a reviewer will raise.

**4.2 Whether to keep a discrete head for evaluation only.** CALM is
likelihood-free, so the HELM paper's MMLU/ARC/HellaSwag numbers — all computed
through `loglikelihood` — become unreachable. A dual-head model (energy head for
training and generation, vocab head used only at eval) would restore
comparability. It costs the 50M head in *parameters* but nothing in training
compute if its gradient is disabled. *Recommendation: decide explicitly rather
than by default.* Losing comparability with the paper's headline table is a real
cost, and this is the only way to keep it.

**4.3 Patch size.** CALM uses K=4. Stage 2a's degradation (99.18 → 97.80 → 88.28)
partly reflects a 24-token sequence leaving only 6 positions at K=4, which will
not apply at 2048 tokens — but K is now a hyperparameter with a measured cost
curve, not a free win. *Recommendation: K=4 to match CALM, with K=2 as the
fallback if quality disappoints.*

**4.4 Hyperbolic latent.** Deferred, and `gmvae/ASSESSMENT.md` explains why: the
elegant part (CALM's `(mean, log_std)` *is* a point on the Gaussian manifold)
does not carry the energy score with it, because KL is not a metric. Stage 3.

---

## 5. Work breakdown

**Phase 1 — promote to library code (no GPU).** The six modules above, with
tests: shape and gradient tests for the patch embedding, the bit-identity test
for `energy_score` that already exists, a BrierLM test against the closed form,
and an end-to-end "loss decreases on a fixed batch" smoke test. This is the
existing repository standard and it is what makes the GPU run debuggable.

**Phase 2 — the autoencoder.** Download CALM's released checkpoint, confirm
`vocab_size=128256` and `patch_size=4`, and measure reconstruction on **HELM's
own corpus** (Wikipedia, Llama-3.1 tokenizer) rather than the Pile it was trained
on. Stage 0 could not do this — Hugging Face is blocked from the development
machine. If transfer is poor, an autoencoder training run is back on the
schedule, and Stage 2b's budget curve is the evidence that it would work.

**Phase 3 — baseline discrete HELM on BrierLM.** Before training anything new.
Without this number, a HELM-CALM BrierLM score means nothing.

**Phase 4 — train HELM-CALM at the 1B preset, K=4.** lr 1e-3, β=1.0, the
optimizer split from §2, MiCE balancing hyperparameters re-tuned for K-fold fewer
routing decisions.

**Phase 5 — measure.** BrierLM against the Phase 3 baseline, and the parked
efficiency benchmark (`PARKED.md`) at the same shape.

---

## 6. What could still kill it

Ordered by how likely they are to matter, and none of them is ruled out by
anything measured so far.

1. **Patching may blur what HELM is for.** HELM's thesis is that hyperbolic
   geometry matches the *token-level* hierarchy of language. CALM compresses four
   tokens into one vector. If the hierarchy lives below the patch boundary, HELM's
   advantage over a Euclidean baseline could evaporate — leaving a model with
   CALM's efficiency and no reason to be hyperbolic. **Attempted and unresolved**
   — see `HIERARCHY.md`. A synthetic tree-structured language was built to test
   it; the instrument turned out not to detect hierarchy at all, because discrete
   HELM itself shows almost none on that task (recovery +0.06 at 98.79% accuracy)
   while its representations become *less* tree-like over training. The risk
   stands, and `HIERARCHY.md` sets out what a working test would need.
2. **The energy score's variance grows with latent dimension.** All of Stage 1
   and 2a used a 32-dimensional latent. CALM uses 128. The estimator averages
   `num_samples=8` draws; whether 8 remains adequate at 128 dimensions and at a
   realistic vocabulary is untested, and the sample count multiplies head
   activation memory directly.
3. **MiCE routing at patch granularity.** K-fold fewer routing decisions, each
   over a semantically broader unit. Load balancing was tuned per token; expert
   collapse is a plausible failure mode and the auxiliary-loss-free bias update
   (fixed in this port, dead upstream) has not been exercised in this regime.
4. **A second evaluation stack is a real cost.** Phase 3 exists because BrierLM
   is not comparable to anything in the HELM paper. Budget it as work, not as a
   flag.
