# Learned sparsity: research and experiment design

The idea under test: **let the model learn its own attention sparsity by
optimising an objective, and compare that against dense MHA and MoE MHA.**

`CRITICALITY.md` argued that attention is already a dense graph annealed by
softmax, with a temperature as its control parameter. T2 made that temperature
learnable and found it worth nothing. This document explains why that result
does **not** settle the sparsity question, what the literature says the right
mechanism is, and how to test it.

---

## 1. Why T2's null does not carry over

T2 made `beta` (the attention temperature) learnable. It moved reproducibly to
0.934x the standard `1/sqrt(d)` and bought no perplexity.

But **beta cannot create sparsity**. Softmax has full support by construction:
for any finite logits and any temperature, every token receives a strictly
positive weight. Rescaling the logits changes how concentrated the distribution
is, never which tokens are in it. T2 tested *temperature*; sparsity is
*topology*. They are different knobs and the null on one is not evidence about
the other.

## 2. The mechanism: alpha-entmax

`alpha-entmax` (Blondel et al. 2019; Peters et al. 2019) replaces softmax with

    alpha-entmax(z) = argmax_{p in simplex}  <p, z> + H^T_alpha(p)

where `H^T_alpha` is the Tsallis entropy family. The solution has closed form

    alpha-entmax(z) = [(alpha - 1) z - tau * 1]_+^{1/(alpha-1)}

with `tau` a Lagrange multiplier found by bisection. The `[.]_+` is what
matters: it is a ReLU, so **entries below the threshold are exactly zero**.

* `alpha = 1` recovers softmax exactly (dense).
* `alpha = 2` gives sparsemax (piecewise linear, very sparse).
* `alpha` in between interpolates continuously, getting sparser as it rises.

**Adaptively Sparse Transformers** (Correia, Niculae & Martins,
`arXiv:1909.00015`) make `alpha` *learnable per head*, as
`alpha = 1 + sigmoid(a)`, by deriving the Jacobian of entmax with respect to
`alpha`. That is precisely "learned sparsity optimising a metric": one extra
scalar per head, trained by the ordinary LM loss, and it decides how many
tokens that head may attend to.

### What they found

| | DE→EN | JA→EN | RO→EN | EN→DE |
|---|---|---|---|---|
| softmax | 29.79 | 21.57 | 32.70 | 26.02 |
| 1.5-entmax | 29.83 | 22.13 | 33.10 | 25.89 |
| alpha-entmax | 29.90 | 21.74 | 32.89 | **26.93** |

Sparse attention does not hurt and is slightly better. **This is a parity
result, not a breakthrough** -- worth stating plainly, because it sets the
effect size we should expect and therefore the power the experiment needs.
Cost was 75% of softmax throughput; ours measures at 40% (no fused kernel).

### The finding that bears directly on our results

Three things from that paper matter here:

1. **Heads become denser first.** Randomly-initialised `alpha` values *decrease*
   early in training; only after ~1000 steps do some heads turn sparse. They
   read this as "dense attention is preferable while the model is uncertain".
2. **Decoder self-attention prefers denser attention than encoder
   self-attention.** They speculate that autoregressive attention cannot afford
   to zero out tokens, having fewer of them available in the first place.
3. Encoder self-attention converges to a *bimodal* distribution of `alpha` --
   some heads nearly sparsemax, others near softmax. Uniformly sparse
   transformers appear to be suboptimal.

Our model is **decoder-only**. So the paper's own evidence predicts the
smallest benefit exactly where we are testing.

And this converges with our own result: T2's learned `beta` moved to 0.934x,
i.e. *flatter*, i.e. *denser*. A quick 40-step probe of learned `alpha` in our
model moves 1.500 -> 1.490, also denser. **Two independent mechanisms, in two
independent codebases, both push autoregressive attention toward density.**
That is now a real pattern rather than a one-off, and it is the most concrete
prediction available to test.

## 3. Where sparsity is actually supposed to help

`Long-Context Generalization with Sparse Attention` (`arXiv:2506.16640`) gives
the mechanism: as sequence length grows, non-informative tokens accumulate
attention mass, causing *dispersion* and representational collapse. Exact zeros
prevent that accumulation.

**The benefit is therefore length-dependent**, and at our `seq_len = 128` there
is very little dispersion to prevent. This is the single biggest threat to the
experiment's informativeness and is addressed in the design below.

## 4. The modern line, for context

Learned sparsity is now mainstream, and the trend is away from fixed top-k:

* **NSA** (`arXiv:2502.11089`, DeepSeek) -- natively trainable hierarchical
  block sparsity, hardware-aligned.
* **SeerAttention** (`arXiv:2410.13276`) -- learns block-level sparsity via a
  gating mechanism rather than a predefined pattern.
* **DashAttention** (`arXiv:2605.18753`) -- replaces NSA's top-k block
  selection with alpha-entmax, explicitly because top-k assumes a *fixed*
  number of relevant tokens and blocks gradient flow to the selection step.
* **AdaSplash** (`arXiv:2502.12082`, `2604.15180`) -- makes entmax competitive
  with fused softmax kernels.

Note what DashAttention's argument implies for MoE: **top-k routing has the
same two defects** -- it fixes the number of experts per token in advance and
is not differentiable in the selection. An entmax router is the obvious
analogue, and is a natural follow-up if the attention result is positive.

## 5. Design

### Arms: a 2x2, paired on the axis that can be paired

|  | softmax attention | alpha-entmax attention |
|---|---|---|
| **dense FFN** | MHA baseline | learned-sparse MHA |
| **MoE FFN** | MoE MHA baseline | learned-sparse MoE MHA |

This answers the question directly: learned sparsity against both MHA and
MoE MHA.

**Pairing.** Within a fixed FFN type, the softmax and entmax arms share every
weight tensor bit-identically at a given seed (verified: 24 of 24 shared
tensors equal; the entmax arm's only extra parameters are the per-head
`alpha_logit`). The data stream seed is fixed and there is no dropout. So the
softmax/entmax contrast is **paired**, exactly as in T3 -- which is what made a
1% effect measurable there when T2's unpaired design needed ~79 seeds.

The dense/MoE contrast **cannot** be paired (different parameter counts). It is
reported, but the design does not depend on resolving it, and T2's null on that
axis should be assumed to persist.

### Measurements

Beyond perplexity, all already instrumented:

* `alpha` per head per layer -- where learned sparsity settles, and its spread;
* `zero_frac` -- the fraction of allowed keys receiving **exactly** zero weight.
  This is the quantity `beta` could not move, and it is the direct evidence of
  whether learned sparsity did anything at all;
* `participation_frac`, `entropy_norm` -- comparable to T2/T3, so we can ask
  whether entmax reaches an operating point softmax cannot;
* `load_balance`, `dead_experts` for the MoE arms.

### Registered predictions

**P1 (main, paired).** alpha-entmax gives parity or a small gain over softmax,
not a large win. Basis: the source paper's own numbers are +0.11 BLEU on
WMT14. **If we observe a large gain, be suspicious of a bug, not delighted.**

**P2 (direction of alpha).** Learned `alpha` falls from its 1.5 initialisation
toward 1 (denser). Basis: the paper's decoder finding, its early-training
trajectory, our T2 `beta` result (0.934x), and a 40-step probe already showing
1.500 -> 1.490. This is the strongest prediction here and the cheapest to
check.

**P3 (interaction).** No interaction between attention sparsity and MoE
routing sparsity. They reduce different graphs -- token-to-token versus
token-to-expert -- so there is no mechanism for them to interact. A measured
interaction would be a genuine surprise and worth chasing.

**P4 (the honest null).** At `seq_len = 128` there is little attention
dispersion to prevent, so the mechanism by which sparsity is supposed to help
is largely absent. A null on P1 is therefore **weak evidence** and must not be
reported as "learned sparsity does not help" -- only as "not at this context
length".

### Budget

Measured, uncontended, per 500-step run:

| arm | ms/step | 500 steps |
|---|---|---|
| dense / softmax | 196 | 1.6 min |
| dense / entmax | 500 | 4.2 min |
| moe / softmax | 237 | 2.0 min |
| moe / entmax | 547 | 4.6 min |

All four arms = 12.4 min per seed. **5 seeds = 62 min; 4 seeds = 50 min.**

### The follow-up that tests the mechanism

If P1 comes back null, the informative next step is **not** more seeds -- it is
the length axis, because that is where the claimed mechanism lives. Run the
paired softmax/entmax contrast at `seq_len` 128 / 256 / 512 and test whether
the entmax advantage *grows with length*. That is a trend test across three
points, analogous to T3's head_dim trend, and it distinguishes "sparsity does
not help" from "we tested it where it cannot help".

Attention cost scales as `n^2`, so seq 512 is ~16x the attention time of 128;
this needs its own budget and probably a smaller model or fewer steps.
