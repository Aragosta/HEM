# Results

## Reproduction gate — WikiText-2, byte level, tier 1

The gate asks one thing before anything else runs: **does HELM's hyperbolic
advantage appear in our hands at all?** If it does not, no CALM-column
difference can be attributed to geometry, and the suite refuses to report an
interaction rather than producing a number that looks like an answer.

Setup: WikiText-2 official splits, byte level, 12,000 steps = 12,288,000 tokens =
1.14 epochs, two seeds, parameters matched to −0.4%.

| cell | top-1 | BPB | brier_1 | **effective rank** | NaN |
| --- | --- | --- | --- | --- | --- |
| `helm_discrete` s0 | 58.66% | 1.9541 | +0.219 | 3.4 | 0 |
| `helm_discrete` s1 | 58.91% | 1.9514 | +0.188 | 4.1 | 0 |
| `euclid_discrete` s0 | 58.89% | 1.9529 | +0.258 | 18.9 | 0 |
| `euclid_discrete` s1 | 58.70% | 1.9767 | +0.211 | 19.5 | 0 |
| *bigram floor* | *32.22%* | *3.3866* | — | — | — |

```
geometry effect: -0.02%  (seed sd 0.18%)
REPRODUCTION GATE FAILED
```

### What is solid

**Both models genuinely learned.** BPB ≈1.95 against a bigram floor of 3.39,
top-1 58.8% against 32.2%, positive brier scores (real signal, not the mode
collapse that made every earlier CALM row negative), zero non-finite steps, seed
spread 0.18%. This is the first point in the project where a cell cleared a
floor set by someone other than us.

**There is no geometry effect here.** −0.02% against a 0.18% seed spread is a
tie. Not a weak effect — an absent one.

### The finding the accuracy column hides

| | top-1 | BPB | effective rank |
| --- | --- | --- | --- |
| HELM | 58.79% | 1.953 | **3.8** |
| Euclidean | 58.80% | 1.965 | **19.2** |

**HELM reaches identical quality on ~5× fewer representational directions.**
That is consistent with what hyperbolic space actually does — embed hierarchy
compactly — and it is the first mechanistic result about the geometry in this
project rather than another repair. It does not convert into better prediction
at this scale, and it costs 2.8× the wall-clock per step (165.7 ms against
59.8 ms at matched parameters).

Whether a 5× more compact representation is worth having is a different question
from whether it predicts better, and this suite was not built to answer it.

### Why this is not evidence against HELM

**Corrected, and the first reason is now the important one.** HELM's published
result stands: HELM-MiCE consistently outperforms 1B DeepSeek-V3, with gains up
to 4% over LLaMA/DeepSeek architectures, and it *"always achieve[s] higher
accuracy on the more difficult reasoning benchmarks, namely MMLU and
ARC-Challenging"*.

**We measured the wrong quantity.** Bits-per-byte and next-byte accuracy are
perplexity-family metrics. HELM's claim is about multiple-choice reasoning
accuracy. A tie on BPB is entirely compatible with a 4% MMLU gain, so this table
does not bear on HELM's claim at all -- and no amount of extra compute at this
setup would have made it bear on it.

Two reasons, and the second is a design mistake of mine rather than a limit of
the hardware.

**Scale.** 449,496 parameters on 12M tokens. HELM's published results are at
120M and 1B on far more data. Architecture comparisons at 450K are not known to
predict anything at 120M.

**Byte level is probably the wrong setting to test HELM in.** HELM's thesis is
that *token* embeddings carry hierarchical, power-law structure that hyperbolic
space embeds with low distortion. 256 byte values have essentially no such
structure. Byte level may remove precisely the property the geometry exists to
exploit — which would make this null partly an artefact of the tokenization,
not only of scale.

Bytes were chosen because HELM's Llama-3 tokenizer is not fetchable in this
environment and bits-per-byte is tokenizer-free, which is right for
comparability across models that do not share a vocabulary. It is the wrong
choice for testing *this particular claim*, and that should have been reasoned
about before the run rather than after it.

### What follows

The GPU tier is now required rather than optional, and needs two changes, not
just more compute:

1. **A real tokenizer**, so token-level hierarchy exists to be exploited.
2. **Scale** — 120M+, WikiText-103, the setting HELM's own results come from.

Until then the honest statement is: *at 450K parameters on byte-level
WikiText-2, HELM and a matched Euclidean transformer are indistinguishable in
quality, HELM is 2.8× slower, and HELM's representation is 5× more compact.*


## Data hierarchy — is there a patch hierarchy for CALM to have?

HELM's premise is about **token** hierarchy. CALM predicts patches, so the
premise only carries over if patches inherit the structure. Every earlier
attempt in this repository to answer that ran through a trained model, where the
answer is confounded with capacity, optimisation and our own bugs. This asks the
corpus directly: co-occurrence graph, positive PMI, Gromov's delta -- the
standard justification for hyperbolic embeddings.

WikiText-2, word level, 2M words, graph over the 400 most frequent units.

| construction | delta | linked |
| --- | --- | --- |
| tokens (K=1) | **0.1954** | 82.8% |
| tokens, word order shuffled | **0.1275** | 92.9% |
| K-gram atoms (K=2) | 0.1840 | 44.3% |
| K-gram atoms (K=4) | *vacuous* | **1.1%** |
| K-gram atoms (K=8) | *vacuous* | 0.7% |
| aggregated profiles (K=1) | 0.0022 | 100% |
| aggregated profiles (K=2) | 0.0037 | 100% |
| aggregated profiles (K=2, scrambled) | 0.0028 | 100% |
| aggregated profiles (K=4) | 0.0084 | 100% |
| aggregated profiles (K=4, scrambled) | **0.0072** | 100% |
| aggregated profiles (K=8) | 0.0099 | 100% |
| aggregated profiles (K=8, scrambled) | **0.0121** | 100% |

### Two established results

**Token-level hierarchy is real.** delta 0.1954 against 0.1275 for the same
construction on shuffled word order. HELM's core premise holds on this corpus,
confirmed without a model in the loop -- the first independent check of it here.

**Patches are not atoms.** Raw K-gram co-occurrence is unusable past K=2: at K=4
only 1.1% of pairs are linked, because the 400 most frequent 4-grams essentially
never co-occur. Whatever "patch hierarchy" could mean, it cannot mean patches
having tree-like relationships among themselves *as units*. It would have to come
from aggregating token structure -- an argument for CALM's `embed_proj` design
and against any patch-vocabulary alternative.

### One negative result, from a control that killed a finding

The aggregate construction produced a clean monotone series -- delta 0.0022,
0.0037, 0.0084, 0.0099 for K = 1, 2, 4, 8 at 100% linked -- which reads as
"patches are 3.9x less tree-like than their tokens" and would have been the
`HIERARCHY.md` worry confirmed.

**It is an artefact.** Building patches from K *randomly sampled* tokens instead
of K consecutive ones -- identical averaging, adjacency destroyed -- reproduces
the same growth, and at K=8 exceeds it. Averaging vectors concentrates them
toward the centroid regardless of what they are, and a monotone trend in K is
exactly what that looks like.

So **there is no evidence here that patching flattens hierarchy.** The
real-versus-scrambled gaps are around 15% and flip sign across K: noise at one
seed. This construction cannot detect a patch-hierarchy effect, because the
averaging artefact swamps whatever signal exists.

### What remains open

The live hypothesis is the one the data cannot reach: that hierarchy lives in a
hyperbolic backbone's **hidden states** over patches, even when neither the raw
statistics nor CALM's Gaussian-regularised latent are tree-like. That needs a
trained model with delta measured on activations -- `probes.hierarchy_flattening`
-- and it needs a geometry effect to exist in the first place, which the
reproduction gate says it does not at 450K parameters on bytes.

### Method note

Three constructions in this table were nearly reported as findings before a
check invalidated them: the K=4 K-gram delta (empty graph), the aggregate
K-trend (averaging artefact), and, in the suite generally, BrierLM's geometric
mean (reads 0.0000 for every model at byte level). In each case the invalidating
quantity was already in the output. `MIN_LINKED`, the scrambled control and the
per-order Brier reporting exist because of those, not in anticipation of them.

## T0 — does HELM's advantage exist in a metric CALM could share?

**The question.** HELM reports its advantage only as multiple-choice accuracy,
scored by the harness picking the choice "with the highest likelihood value"
(HELM, Appendix C.3), and reports **no perplexity anywhere**. CALM's head is an
implicit sampler with no likelihood, so every number HELM reports is one a
HELM-CALM cannot produce. Before building the integration, ask a cheaper
question with no CALM in it: does HELM beat a matched Euclidean model on
**perplexity**, a metric CALM can be compared on through BrierLM?

**Setup.** WikiText-2 official splits, BPE-16000 trained on the train split
(LLaMA-3.1's tokenizer is not fetchable here), 8.96M vs 9.03M parameters
(+0.7%), 4000 steps = 2.58 epochs, seq 192, two seeds. Both arms **dense**
(HELM-D via `n_dense_layers`, not MiCE) so the comparison is within an
architecture family, as the paper's own comparisons are. Both arms **rotary**.
HELM's manifold parameters under `RiemannianAdam` with 3% warmup, verified
on-manifold throughout (`manifold_err` ~3e-07). **Each arm at its own swept
learning-rate optimum**, both bracketed by interior minima over nine points.

### Result

| arm | seed | perplexity | top-1 | brier_1 | ms/step |
| --- | --- | --- | --- | --- | --- |
| HELM-D | 0 | 264.54 | 20.54% | +0.109 | 393.6 |
| HELM-D | 1 | 273.75 | 20.00% | +0.063 | 451.8 |
| Euclidean | 0 | **85.64** | **27.91%** | +0.172 | 276.9 |
| Euclidean | 1 | **86.00** | **27.90%** | +0.109 | 273.6 |

```
perplexity   HELM 269.15 (sd 6.51)   Euclidean 85.82 (sd 0.26)
             difference -183.32, i.e. 3.1x worse, at 28x the seed noise
top-1        HELM 20.27%             Euclidean 27.91%   difference -7.64%
```

**HELM is worse on every shared metric**, by margins far outside seed noise.

### The decile table is the interesting part

Perplexity by token-frequency decile, 0 = most frequent, 9 = rarest:

| decile | HELM | Euclidean | HELM worse by |
| --- | --- | --- | --- |
| 0 | 7.84 | 5.53 | 42% |
| 1 | 6.69 | 5.31 | 26% |
| 2 | 13.51 | 8.83 | 53% |
| 3 | 38.71 | 18.39 | 111% |
| 4 | 161.19 | 50.39 | 220% |
| 5 | 797.55 | 174.76 | 356% |
| 6 | 2366.21 | 394.98 | 499% |
| 7 | 5855.37 | 969.26 | 504% |
| 8 | 14822.78 | 2600.88 | 470% |
| 9 | 42521.62 | 4214.11 | **909%** |

**HELM is worse everywhere and monotonically worse toward the tail** — 42% on
the most frequent decile, 909% on the rarest.

That is the exact opposite of the claimed mechanism. HELM's own case study
(Table 3) locates the geometry's contribution in the tail: generic words cluster
at small norm, specific words at large norm, giving "better separation of
long-tail tokens". Here the tail is where the hyperbolic model does worst.

### Two readings, and the honest one is a mix

**Reading A — the geometry does not help language modelling.** The advantage is
absent on perplexity, and absent hardest where it is claimed to live.

**Reading B — HELM is undertrained, and the geometry is why.** The two arms want
learning rates **16x apart**: HELM's optimum is 2e-4 and it diverges above that
under `RiemannianAdam`, while the Euclidean control's optimum is 3.2e-3. At 2.58
epochs, being confined to a 16x lower rate means far less effective progress,
and **rare tokens are exactly where undertraining shows first** because they are
seen fewest times. The monotone tail degradation is the signature of an
undertrained model, not necessarily of bad geometry.

These are not alternatives so much as a chain: **the geometry imposes an
optimization constraint, the constraint costs training progress, and the cost
lands hardest in the tail.** On that reading the 3.1x is real but is a statement
about hyperbolic *optimization* at this scale rather than about hyperbolic
*representation*.

### What this does and does not establish

**Does not contradict HELM's paper.** HELM reports no perplexity. Its gains are
1-2 points of near-chance multiple-choice accuracy (ARC-Challenge is *below*
chance for every model in its Table 1; MMLU within a point of chance for all
six). A model can be worse at next-token prediction and better at
likelihood-ranked MCQ.

**Scale.** 9M parameters on 6M tokens against the paper's 115M on 5B — roughly
800x less data. A null at this scale is weaker evidence than a positive result
would be.

**Model.** This is HELM-D, not HELM-MiCE. The paper's headline model is the
mixture-of-curvature version and it reports MiCE beating D. Making both arms
dense was necessary for a family-matched comparison, and it costs HELM its best
configuration.

### Recommendation

**Do not build HELM-CALM on the current evidence.** Not because the geometry is
disproven, but because:

1. the only metric a HELM-CALM could be evaluated on shows the geometry
   **losing by 3.1x**, so there is nothing measurable for it to inherit;
2. the benchmarks where HELM's advantage is established are likelihood-scored,
   which CALM structurally cannot do;
3. CALM's energy score allocates training signal in proportion to data density,
   i.e. away from the tail — and the tail is both where HELM's mechanism is
   claimed to act and where it is measurably weakest here.

**What would change this**, in order of cost: rerun T0 with **HELM-MiCE against a
Euclidean MoE** at 120M with a real tokenizer on 5B tokens, which is the paper's
own setting; and report **BrierLM** there, since that is the one metric both a
discrete and a continuous model can share. If HELM's advantage appears in
BrierLM at that scale, the integration becomes measurable and worth building.
Until then it is not.
