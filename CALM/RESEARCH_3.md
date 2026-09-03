# Reading the actual papers

Exa and Firecrawl are connected, so this pass **read HELM's full text** rather
than abstracts. It supersedes parts of `RESEARCH_2.md`, which was built from
search snippets. Where the two disagree, this one is right.

## 1. HELM's real numbers

Table 1 of [HELM](https://arxiv.org/abs/2505.24722), NeurIPS 2025. Trained on
~5B tokens of English Wikipedia, LLaMA-3.1 tokenizer, 128K vocab, seq len 2048.
The second figure in each cell is **points above or below random chance** for
that benchmark, which the paper does not report.

| model | size | CSQA | HellaSwag | OBQA | MMLU | ARC-C | avg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LLaMA | 115M | 20.9 (+1) | 25.1 (+0) | 25.4 (+0) | 23.4 (−2) | 21.0 (−4) | 23.2 |
| HELM-D | 115M | 20.3 (+0) | 25.9 (+1) | 27.1 (+2) | 25.6 (+1) | 21.4 (−4) | **24.1** |
| DeepSeekV3 | 120M | 19.3 (−1) | 25.3 (+0) | 24.0 (−1) | 23.9 (−1) | 22.2 (−3) | 22.2 |
| HELM-MiCE | 120M | 19.7 (−0) | 25.9 (+1) | 27.7 (+3) | 24.4 (−1) | 23.2 (−2) | **24.1** |
| DeepSeekV3 | 1B | — | 26.2 (+1) | 27.4 (+2) | 23.6 (−1) | 22.7 (−2) | 23.9 |
| HELM-MiCE | 1B | 19.8 (−0) | 26.5 (+2) | 28.4 (+3) | 25.9 (+1) | 23.7 (−1) | **24.9** |

**HELM does beat its Euclidean counterparts, consistently.** MiCE beats
DeepSeekV3 on all five benchmarks at 120M and at 1B; the ablations show it also
beats constant-curvature MiCE, learned positional encoding, and full hyperbolic
attention. The paper reports ±0.1–0.5 std over three runs at 100M, and Table 9
shows the 1B gap holding across three training checkpoints. Within its own
setup, the effect is reproducible.

**But every model in the table is at or below random chance.** ARC-Challenge is
*below* chance for all six. MMLU is within ±1 point of chance for all six.
CommonsenseQA and HellaSwag are at chance. Only OpenbookQA is clearly above, and
the paper notes it converted OBQA's answer choices to letters *"to make up for
the relatively smaller model and training dataset sizes."*

At 5B training tokens this is expected — these models do not have MMLU ability.
The consequence matters though: **the reported gains are differences in
near-chance accuracy.** A consistent 1–2 point edge across five benchmarks and
three checkpoints is a real signal of *something*, but at chance level that
something can be answer-order bias, length or calibration effects, or
likelihood-scaling differences rather than reasoning. The paper's own framing —
*"better reasoning capability"* — is not established by numbers in this range.

**And no perplexity is reported anywhere in the paper.** The standard
language-modelling metric is absent, so there is no evidence either way on
whether the geometry improves next-token prediction. That is worth knowing
because it means **our BPB tie does not contradict HELM at all** — HELM never
claimed BPB. `suite/RESULTS.md`'s conclusion stands but for this reason, not the
one originally given.

**Costs, from the paper.** HELM is **1.5–1.8× slower to train** (72h vs 40h for
1B on 4×A800), 1.55× runtime and 1.11× memory per iteration.

**And a detail that matches our code finding.** For the 1B model, training true
hyperbolic embeddings *"cause[d] training instability"*, so they **only train the
space-like dimension and compute the time-like dimension afterwards**. That is
precisely the structure we measured in `LorentzLinear`: a Euclidean space part
with a derived time coordinate. At 1B, HELM's own embedding is Euclidean-plus-
derived-coordinate by necessity.

## 2. The evaluation problem is now confirmed verbatim

From HELM's Appendix C.3:

> We use the Language Model Evaluation Harness library for all evaluations,
> where the framework **prompts the models with the answers choices to each
> question and picks the one with the highest likelihood value.**

Every number HELM reports is produced by comparing **likelihoods**. CALM's head
is an implicit sampler with **no likelihood** — that is its defining property and
the reason its own paper reports BrierLM instead of perplexity.

So: **the entire evidential basis for HELM's advantage is a measurement a
HELM-CALM cannot perform.** This is not a limitation of our implementation. It
is structural, and no amount of compute removes it.

## 3. The energy score's density weighting, confirmed

Verbatim from [Decision-Aware Training for Sample-Based Generative
Models](https://arxiv.org/abs/2607.01171):

> These models are commonly trained with strictly proper scoring rules, such as
> the energy score, which **allocate their training signal in proportion to data
> density**, with no awareness of where forecast errors are most costly.

Set against HELM's own qualitative finding (Table 3) that generic words cluster
at small norm and specific words at large norm — i.e. **the geometry's work is
done in the tail** — this gives the sharpest mechanistic worry available:

**CALM's objective weights training signal by density; HELM's advantage lives in
the low-density tail. The two are pulling in opposite directions.**

This is a hypothesis, not a measurement. But it is now sourced on both sides.

## 4. So: is the integration worth it?

**The honest assessment, having read the numbers.**

The case rests on a 1–2 point average edge, measured entirely at near-chance
accuracy, on benchmarks scored by a likelihood that CALM cannot produce, in a
model that trains 1.5–1.8× slower. Every one of those clauses is from HELM's own
paper.

That is not a reason to dismiss HELM — its result is real within its setup, the
ablations are careful, and the semantic-hierarchy case study (generic words at
smaller norm than specific ones, which does *not* hold for DeepSeekV3) is a
genuine qualitative finding that our own δ measurement on WikiText-2
independently supports.

It is a reason to be sceptical that the effect **survives composition with
CALM**, because:

1. the evidence for it cannot be reproduced in a CALM model (§2);
2. the mechanism it is attributed to lives where CALM's objective is weakest (§3);
3. the effect size, on the only evidence available, is small relative to the
   noise floor at chance-level accuracy.

**What would change my mind, in order of cost.**

**T0 — Establish HELM's advantage on a metric CALM can share.** Train HELM and
DeepSeekV3 at ~120M on the paper's own setup and measure *perplexity* and
*BrierLM*, neither of which the paper reports. If HELM's advantage appears in
BrierLM, the integration becomes testable and worth pursuing. **If it appears
only in likelihood-scored MCQ, the integration is not measurable and should not
be built.** This is the single highest-information experiment available and it
does not require building HELM-CALM at all.

**T1 — Frequency-stratified evaluation.** Split BrierLM and accuracy by token
frequency decile. Tests §3 directly: if HELM's edge is in rare deciles and CALM's
objective flattens it there, the interaction is negative for a known reason.

**T2 — The manifold ablation** (`RESEARCH_2.md` §1). Still worth an hour: replace
`LorentzLinear` with `nn.Linear` and see what moves. HELM's own 1B embedding
concession (§1) suggests the answer may be "less than expected".

**T3 — The 2×2 at 120M with the paper's setup**, only if T0 says the advantage is
measurable in a shared metric.

**My recommendation:** do **T0 before building anything else.** It is a
reproduction study, not an integration, and its outcome determines whether the
integration can be evaluated at all. Building HELM-CALM first and discovering
afterwards that its advantage is unmeasurable would be the expensive version of
the same finding.
