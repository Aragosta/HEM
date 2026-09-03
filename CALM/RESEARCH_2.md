# What to test next, and whether HELM-CALM is worth it

> **Correction (added after the first version).** This document originally leaned
> on our -0.02% null as though it were weak evidence against a hyperbolic
> advantage. It is not evidence at all, and for a sharper reason than scale.
> HELM's published result is real: **HELM-MiCE consistently outperforms 1B
> DeepSeek-V3, with gains up to 4% over the Euclidean architectures used in
> LLaMA and DeepSeek** (NeurIPS 2025). See section 0.5 -- the advantage lives on
> benchmarks we did not measure and, more importantly, on benchmarks a CALM
> model may be unable to measure at all.

A literature pass across hyperbolic deep learning, latent/patch language models,
scoring rules, scaling methodology and representation geometry, plus two code
measurements that came out of it. It ends with a ranked test plan and an honest
answer to "is it worth it".

## 0. What this pass could and could not do

**It could not read papers.** arxiv, Semantic Scholar, NeurIPS, PMLR, ACL
Anthology, OpenReview and Springer are all blocked by this environment's egress
proxy. What follows is built from **abstract-level search results across ~45
papers**, from three full codebases available locally (HELM, CALM, GM-VAE), and
from two measurements run here. Source code is stronger evidence than an
abstract; a synthesised search snippet is weaker than either. Claims below are
marked accordingly.

## 0.5 The metric, not just the scale, was wrong -- and this is the crux

HELM's reported gains are on **MMLU and ARC**, and the paper's own framing is
that HELM *"always achieve[s] higher accuracy on the more difficult reasoning
benchmarks, namely MMLU and ARC-Challenging"*
([HELM](https://arxiv.org/abs/2505.24722), NeurIPS 2025). The advantage is
described as concentrated in the **harder** reasoning tasks, not spread
uniformly.

Our reproduction gate measured **bits-per-byte and next-byte accuracy**. Those
are perplexity-family metrics. Nothing in HELM's claim says perplexity should
move much, and a hyperbolic-vs-Euclidean tie on BPB is compatible with a 4% MMLU
gain. **So the gate did not fail to reproduce HELM; it measured a quantity
HELM's claim is not about.** That is a worse methodological error than running
at 450K parameters, because more compute would not have fixed it.

**And this is the crux for the integration.** MMLU and ARC are scored by
comparing the **log-likelihood** of each candidate continuation. CALM has no
likelihood -- that is the defining property of its implicit-sampler head, and
the reason its own paper reports BrierLM instead of perplexity.

So:

* the benchmarks where HELM's advantage is established are **likelihood-based
  multiple choice**;
* a CALM model **cannot produce likelihoods**;
* therefore **HELM-CALM cannot be evaluated on the evidence HELM's claim rests
  on**, at least not in the form the claim was made.

This is not a statement that the integration cannot work. It is a statement that
we would have no way to tell whether it worked, using the measurement that made
HELM credible in the first place. A generative workaround exists -- emit the
answer letter and score exact match -- but it is noisier and is *not* how HELM's
table was produced, so the comparison to that table would be broken.

**Any serious plan has to solve the evaluation problem before the modelling
problem.** That reorders the test plan in section 5.

## 1. The finding that best explains our null

**Measured here, not read:** HELM's `LorentzLinear` — the layer the whole
"fully hyperbolic" claim rests on — computes its space part with a plain
Euclidean linear map, exactly:

```
LorentzLinear(x).space  ==  F.linear(x, W, b)      max abs diff 0.0
time coordinate         ==  sqrt(|space|^2 + c)     corr with |space| 0.99995
```

So a stack of these is: Euclidean linear → append a coordinate that is a
deterministic function of the output norm → Euclidean linear → … The time
coordinate carries **no independent information**; it is a derived feature. And
at the head, after `LorentzRMSNorm`, we measured it to be **constant** (std
exactly 0.0000).

The geometry does still enter in four places — the manifold-constrained
embedding, the Lorentzian inner product in attention, `LorentzRMSNorm`'s
normalisation, and the derived coordinate being visible to the next layer. But
the *linear layers themselves are Euclidean*.

This has an independent echo in the literature. The standard critique of
tangent-space hyperbolic networks is that chained `expmap`/`logmap` pairs are
mutually inverse and **"effectively cancel out, reducing the sequence to
approximately the original Euclidean transformation"**
([Hyperbolic Graph Learning review](https://arxiv.org/html/2202.13852v3),
[Fully Hyperbolic NNs](https://arxiv.org/pdf/2105.14686)). HELM avoids that
particular failure — it never takes tangent-space detours — but arrives
somewhere adjacent by a different route.

**This is the first mechanistic explanation for our −0.02% geometry effect that
does not appeal to scale.** It is also directly testable (§5, T1).

## 2. Our scale was below where anything is measurable

[Small-Scale Experiments: Are We There Yet?](https://arxiv.org/abs/2608.11859)
finds scaling laws **unreliable below ~4M parameters**, and identifies the
confound: small models are extremely sensitive to hyperparameters, a sensitivity
that fades with scale, so effects "only emerge on the fully tuned frontier".
Encouragingly, it also finds that **model-centric improvements identified in the
4M–34M range did predict behaviour at larger validation scale**.

Our reproduction gate ran at **449,496 parameters with a single shared learning
rate across all cells**. That is an order of magnitude below the stated
reliability floor, with exactly the confound the paper names. The null is
uninformative about HELM, and I should have known the threshold before spending
the compute.

**Prescription: 4M–34M parameters, with a per-cell learning-rate sweep.** Not
more steps at 450K.

## 3. The sharpest hypothesis: CALM's objective may suppress where HELM helps

Two findings that only matter together.

**Where hyperbolic geometry helps.** Frequent, abstract tokens sit near the
origin; rare, specific tokens sit far out, and the exponential volume growth
gives **"better separation of long-tail tokens for prediction"**
([Hyperbolic LLMs](https://arxiv.org/html/2509.05757v1),
[Hyperbolic Fine-tuning](https://arxiv.org/html/2410.04010v1)). The advantage is
concentrated **in the tail**.

**Where the energy score puts its signal.** Sample-based generative models
trained with strictly proper scoring rules **"allocate their training signal in
proportion to data density"**
([Decision-Aware Training](https://arxiv.org/pdf/2607.01171)).
That is, *away from* the tail.

**So CALM's objective may systematically de-emphasise exactly the region where
HELM's geometry pays off.** If true, the interaction in our 2×2 is negative for
a reason that has nothing to do with patching destroying hierarchy — it is the
scoring rule, not the aggregation.

This is the most valuable hypothesis this pass produced, because it is specific,
mechanistic, and cheap to test: stratify quality by token frequency (§5, T2).

It also retro-explains the byte-level null completely: **256 byte values have no
long tail.** Every byte is frequent. There is no leaf-level structure for
hyperbolic geometry to separate, so no advantage to find — independent of scale.

## 4. What the field has established that bears on "is it worth it"

**Patching works, and dynamic beats fixed.**
[Byte Latent Transformer](https://arxiv.org/abs/2412.09871) matches
tokenizer-based LLMs up to **8B parameters on 4T bytes**, using
**entropy-based dynamic patching** with an average patch size of **four bytes**.
Both entropy patching and simple space-patching **beat MegaByte's fixed
patching** — and fixed-K patching is what CALM does. So CALM's efficiency thesis
is validated by neighbours, but its *patching scheme* is the weaker known
variant.

**CALM has a strong, simpler competitor.**
[Multi-token prediction](https://arxiv.org/abs/2404.19737) gets **3× faster
inference and better quality** with n independent output heads on a shared
trunk — no autoencoder, no energy score, no likelihood-free machinery. At 13B it
solves 12% more HumanEval and 17% more MBPP. Crucially it is **"increasingly
useful for larger model sizes"**. Any case for CALM has to be made against this,
and CALM's added complexity needs to buy something MTP does not.

**Reconstruction fidelity is not the right precondition.** Work on continuous
latent LMs finds that **"a VAE trained solely for token reconstruction can
achieve near-perfect accuracy yet produce latents poorly suited for conditional
denoising"** — the bottleneck is **representation effectiveness**, whether the
latent is *predictable from context*, not whether it round-trips
([Continuous Latent Diffusion LM](https://arxiv.org/abs/2605.06548)).

Our autoencoder hits **99.95%** reconstruction. By CALM's stated criterion that
is a pass. By this criterion it is **untested**, and it is the more relevant one.

**Low effective rank is ambiguous.** High-quality representations have *low
intrinsic dimension* but *high effective rank* — spreading information across
ambient dimensions avoids collapse; intrinsic dimension correlates with
downstream accuracy better than effective rank does
([Shape of Learning](https://aclanthology.org/2024.findings-eacl.58.pdf)).
Our HELM cell had effective rank **3.8** against Euclidean's **19.2** at equal
loss. Under this framing that reads as *possible collapse*, not efficiency —
though equal BPB argues information was preserved. **Intrinsic dimension is the
measurement that disambiguates, and we did not take it.**

**Curvature should be fitted, not assumed.** The recommended practice is to
estimate curvature per dataset from measured δ-hyperbolicity rather than fixing
it. We used `c = 1.0` everywhere. Separately,
[Robust Hyperbolic Learning](https://arxiv.org/html/2405.13979) reports that
curvature learning is unstable without care, and that **Riemannian AdamW** plus
a *smooth* scaling function (rather than clipping) is what makes it work — we
used plain AdamW with a hard retraction and a hard radius clamp.

## 5. The test plan, ranked by information per GPU-hour

**T0 — Fix the evaluation before anything else (design work, no compute)**
HELM's advantage is established on likelihood-scored multiple choice, which CALM
structurally cannot do (§0.5). Decide now which of these the project accepts:
(a) evaluate HELM-CALM generatively on MMLU/ARC by emitting the answer and
scoring exact match, accepting that it is not comparable to HELM's table;
(b) restrict claims to BrierLM and generative benchmarks, and give up on
comparing to HELM's published numbers; or (c) keep a likelihood-bearing head
alongside the CALM head purely for evaluation. **Without one of these, no amount
of GPU produces a number that speaks to HELM's claim.**

**T1 — Is HELM's geometry doing anything? (CPU, ~1 hour, do this second)**
Ablate the manifold from HELM piece by piece: replace `LorentzLinear` with plain
`nn.Linear` (§1 says this is a no-op on the space part), then the Lorentzian
attention inner product, then `LorentzRMSNorm`, then the manifold embedding.
Whichever ablation does *not* change the loss was never contributing. If none
changes it, HELM is Euclidean in effect and the whole integration question is
moot. **Cheapest possible test of the most fundamental assumption, and it needs
no GPU.**

**T2 — Frequency-stratified quality (cheap, high information)**
Report BPB and block accuracy bucketed by token frequency decile, in all four
cells. Tests §3 directly. If HELM's advantage lives in the rare deciles and
CALM's objective flattens it there, that is the mechanism, and it is visible
long before it shows in an aggregate.

**T3 — Latent predictability, not reconstruction (cheap)**
Measure how much of the latent is predictable from context — conditional
variance explained, or the energy score achieved by an oracle that sees the true
next patch. §4 says this, not reconstruction, is CALM's real precondition. We
have never measured it.

**T4 — The 2×2 at 4M–34M with per-cell LR sweep (GPU, the main run)**
The scale §2 identifies as the smallest that predicts. Real tokenizer, so a long
tail exists at all. WikiText-103. Three seeds. This is the run that answers the
original question, and T1–T3 should gate whether it is worth launching.

**T5 — Intrinsic dimension alongside effective rank (free, add to the suite)**
Disambiguates compression from collapse (§4).

**T6 — Curvature fitted from data δ, and Riemannian AdamW (moderate)**
Removes two confounds that currently favour the Euclidean cell.

**T7 — Entropy-based dynamic patching (larger change)**
BLT's finding that dynamic beats fixed. Only worth it if T4 shows the
integration has legs.

## 6. Is it worth it? An honest answer

**The case against, and it is substantial.** Multimodal hyperbolic work reports
performance **"comparable to Euclidean baselines"** — though HELM itself does
better than that, beating 1B DeepSeek-V3 by up to 4% on MMLU and ARC —
compactness and uncertainty calibration are the claimed wins, not accuracy.
Multi-token prediction already delivers CALM's efficiency with less machinery
and *proven* quality gains at 13B. BLT already delivers byte-level patching at
8B with a better patching scheme than CALM's. And §1 suggests HELM's linear
layers may be Euclidean in effect anyway.

**The case for, narrowly.** One combination is genuinely untested and has a
mechanism behind it: hyperbolic geometry's advantage is in the **long tail**,
and patch-based models are exactly where the tail is hardest — a patch of K rare
tokens is rarer still. If HELM's separation of long-tail tokens survives
aggregation, HELM-CALM would help most precisely where CALM is weakest. That is
a real hypothesis, and nobody has tested it.

**What I would actually do.** Run **T1** before anything else. It costs an hour
of CPU and can end the project honestly: if ablating the manifold does not move
the loss, there is no integration to build. If the geometry *is* load-bearing,
run T2 and T3, and only then spend GPU on T4.

The expensive mistake available right now is launching T4 on the assumption that
the geometry works, when a one-hour ablation can check.
