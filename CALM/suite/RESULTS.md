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

---

# T2 — dense MHA vs MoE MHA, and is `1/sqrt(d)` the right temperature?

WikiText-2, BPE-16000, dim 192, 4 layers, head_dim 32, kv_latent 48, seq 128,
**700 steps (0.30 epochs), 2 seeds**, lr 3e-3.

**Read the caveat before the numbers.** This run was capped at one hour. The
learning rate was **not swept** — 3e-3 is an extrapolation from a grid that
never bracketed its optimum — and 700 steps is a third of an epoch. The seed
standard deviation is 1.5–5.0 perplexity while the entire spread between arms
is 1.6. Every perplexity comparison below is therefore inside the noise, and
none of them should be quoted as a result.

| arm | seed 0 | seed 1 | mean | seed sd | total params | active |
|---|---|---|---|---|---|---|
| dense/fixed | 194.41 | 201.54 | 197.97 | 5.04 | 5,248,896 | 5,248,896 |
| dense/learned | 196.17 | 198.35 | 197.26 | 1.54 | 5,248,920 | 5,248,920 |
| moe/fixed | 199.59 | 193.15 | 196.37 | 4.55 | 6,431,616 | 5,251,968 |
| moe/learned | 198.05 | 194.69 | 196.37 | 2.38 | 6,431,640 | 5,251,992 |

Active parameters differ by 0.06% (the router) against a 22% difference in
total parameters, so the FFN comparison is FLOP-matched as intended.

## P1 — routing at matched FLOPs: **not resolvable**

MoE wins the mean by 0.81% (fixed beta) and 0.45% (learned beta). Both are
noise. The paired per-seed differences for the fixed-beta row are **−5.18 and
+8.39** — the sign flips and both magnitudes meet or exceed the seed sd. The
two arms' seeds happen to anti-correlate, which is what noise looks like at
this scale.

Reporting "MoE better by 0.8%" from the means alone would have been an
artefact. The honest statement is: **no measurable difference between routed
and dense FFNs at matched active FLOPs after 700 steps.**

The null is clean rather than broken: `load_balance` is 0.78–0.83 against a
collapse floor of 0.25 for four experts, and zero dead experts in every run.
The router is genuinely spreading tokens.

## P2 — learnable temperature: **fails, and the direction is the informative part**

Perplexity: 0.36% (dense) and 0.00% (MoE). Nothing.

The direction is the result. `beta_ratio` came out **0.924, 0.936, 0.928,
0.947** — mean **0.934** — across four independent runs spanning both FFN
types. The learned temperature settles ~7% *below* `1/sqrt(d)` every time,
with under 2.5% spread. **Flatter, not sharper.**

The registered prediction was the opposite, and the reasoning behind it was
wrong. I argued that because initialisation sits fully in the disordered phase
(`entropy_norm` 1.0000, `participation_frac` 0.9997), the useful direction must
be toward concentration. That confuses where the model starts with how it gets
where it is going.

Because beta is a paired measurement on identical initialisations — the
`learned` arm is bit-identical to `fixed` at step 0 — seed noise cannot explain
this, which is why it is the only trustworthy effect in T2.

## P3 — do the order parameters track perplexity? **No, and the reason is interesting**

| arm | part_frac | entropy | router_ent | load_bal | dead |
|---|---|---|---|---|---|
| dense/fixed | 0.2174 | 0.4097 | — | — | — |
| dense/learned | 0.2112 | 0.4336 | — | — | — |
| moe/fixed | 0.2167 | 0.4313 | 0.5207 | 0.7841 | 0 |
| moe/learned | 0.1985 | 0.3969 | 0.4996 | 0.8079 | 0 |

The rank orders of `participation_frac` and perplexity do not match, so P3 is
unsupported.

But the table says something the prediction did not anticipate. **All four
architectures converge to the same operating point**: `participation_frac`
0.20 ± 0.02, `entropy_norm` 0.41 ± 0.03, from an initialisation at 1.00. Dense
or routed, fixed or learned temperature, every arm lands in the same place.

And the two knobs move *against* each other. The learned arm has a **flatter**
temperature (0.934x) yet ends **more** concentrated (part_frac 0.2174 → 0.2112
dense, 0.2167 → 0.1985 MoE). The q/k weights over-compensated for the relaxed
temperature.

The reading, offered as a hypothesis and not a result: **the operating phase is
an attractor, and beta is redundant with the q/k weights as a route to it.**
The model has several ways to set how concentrated its attention is — the scale
of q and k, their alignment, and the temperature — and it will reach its target
phase through whichever are available. That is a direct mechanistic explanation
for why a free per-head scalar buys no perplexity: it is not adding a degree of
freedom, it is duplicating one.

## What this costs the larger argument

C2 was the cheap route to re-opening T0. If HELM's 269-vs-86 collapse were a
temperature bug from inheriting `1/sqrt(d+1)` off a different inner-product
scale, a learnable beta would fix it for free. **P2 says a learnable beta is
worth nothing at matched geometry**, so that route is much weaker.

It is not closed. The redundancy argument above says beta does not matter *when
the weights can compensate*. The Lorentz arm's inner products have a genuinely
different magnitude scale, and T0's hyperbolic arm was also confined to a 16x
lower learning rate — so the weights there may not have been free to compensate.
That is the `head_geometry x beta_mode` cross, which is now worth running for a
sharper reason than the one originally proposed.

## Methodological finding, which matters more than any arm here

**The order parameters survived the budget cut and the perplexity comparisons
did not.** `participation_frac` reproduced to three significant figures across
seeds (0.2173 vs 0.2176 on dense/fixed) while perplexity on those same two runs
moved 3.6%. Instrumentation of internal state was an order of magnitude more
sensitive than the loss at the same compute.

The corollary for anything run next: **more seeds beat more steps.** Four to six
seeds at 700 steps would resolve P1 and P2; two seeds at 2000 steps would not.

---

# T3 — euclidean vs Lorentz head geometry, PAIRED

Same setup as T1/T2 (WikiText-2, BPE-16000, dim 192, 4 layers, kv_latent 48,
seq 128, 500 steps, lr 3e-3), but the arms are compared **per seed** rather than
by their means. The lift to the hyperboloid is a function, not a layer, so both
arms have identical parameter names, shapes and values at a shared seed
(6,135,936 each, verified bit-identical by `assert_paired`), see identical
batches, and use no dropout. The per-seed difference therefore cancels the
initialisation variance that made T1 and T2 unresolvable.

| head_dim | seed | euclidean | lorentz | diff |
|---|---|---|---|---|
| 16 | 0 | 230.19 | 237.01 | +6.82 |
| 16 | 1 | 230.28 | 236.28 | +6.00 |
| 16 | 2 | 226.92 | 226.52 | −0.40 |
| 16 | 3 | 231.30 | 230.77 | −0.52 |
| 16 | 4 | 229.44 | 230.89 | +1.45 |
| 16 | 5 | 226.71 | 230.25 | +3.54 |
| 32 | 0 | 231.64 | 236.72 | +5.08 |
| 32 | 1 | 232.20 | 235.91 | +3.71 |
| 32 | 2 | 228.62 | 228.94 | +0.33 |
| 32 | 3 | 233.78 | 238.00 | +4.22 |
| 32 | 4 | 232.93 | 231.16 | −1.77 |
| 32 | 5 | 231.23 | 232.66 | +1.44 |

| | n | mean | p (paired t) | 95% CI | positive |
|---|---|---|---|---|---|
| head_dim 16 | 6 | +2.81 | 0.081 | [−0.50, +6.13] | 4/6 |
| head_dim 32 | 6 | +2.17 | 0.099 | [−0.59, +4.93] | 5/6 |
| **pooled** | 12 | **+2.49** | **0.010** | **[+0.72, +4.27]** | 9/12 |

## Result 1 — a real ~1.1% penalty

Lorentz head geometry costs **1.08% perplexity**, p = 0.010, CI excluding zero.
This is the only architectural comparison in the suite that is both resolved
and free of confounds: T0's effect was larger but its arms differed in
optimizer and learning rate, while here the arms differ *only* in whether the
attention score is a dot product or a Minkowski inner product.

The sign test disagrees (9/12, p = 0.146), so the effect is carried by
magnitude rather than consistency. Three seeds went the other way. The honest
statement is "worse on average by ~1%, with individual runs varying in sign",
not "reliably worse".

## Result 2 — the mechanism is NOT dimension efficiency

`WHY_HYPERBOLIC.md` predicted the hyperbolic effect should be largest at small
head_dim and shrink as head_dim grows, because the FlyWire evidence puts the
hyperbolic/Euclidean crossover at d=8-16. Measured: +2.81 at head_dim 16,
+2.17 at head_dim 32, **interaction p = 0.708**. Flat.

Per the interpretation rule fixed before the run, a penalty flat in head_dim
indicates a **fixed cost of the lift** — the time coordinate `sqrt(|x|^2 + c)`
spending representational capacity on `|q|`, which the network already encodes
— rather than a dimension-efficiency effect. The connectome argument does not
transfer to attention.

## What this settles

With T0 (residual stream, 3.1x worse) this closes the line of enquiry:

* hyperbolic geometry in the residual stream — large loss;
* hyperbolic geometry in per-head attention space, the one regime
  `WHY_HYPERBOLIC.md` identified as favourable — small but real loss, and the
  predicted mechanism is absent.

No configuration tested shows hyperbolic geometry helping. **Recommendation:
stop pursuing the HELM side of the CALM-HELM integration.**

## Caveats

5M parameters, 500 steps (0.21 epochs), unswept lr 3e-3, WikiText-2, one seed
family. A 1% effect at this scale could change sign at 120M parameters or at
convergence, which is where HELM's own claims live. The flat head_dim trend is
the more scale-robust half of the finding: it is evidence about *mechanism*,
and it says the proposed mechanism is not operating.

## Method note

Three hypotheses died during this run, each with its falsification condition
registered before the data that killed it:

1. **bimodality** of the differences — died when seed 4 landed in the gap;
2. **seed-correlation** across head_dims (r = 0.98 at n=3) — died when seed 3
   inverted at head_dim 32, r falling to 0.60;
3. **my own decision rule** `|mean|/sem > 2` — the normal approximation, wrong
   at n=6 where the critical value is t(0.975,5) = 2.571. Applying it as
   written would have declared head_dim 16 resolved at p = 0.081. Now fixed in
   the script to print a t statistic against the correct critical value.

The same discipline caught an earlier error in the opposite direction: the
euclid/lorentz gap was dismissed as "0.7%, within noise" by comparing arm
means, when four paired comparisons across four learning rates had already gone
the same way. Pairing was available the whole time and would have made T1
answerable at its original budget.

---

# T5 — learned sparsity vs MHA and MoE MHA: does the MECHANISM matter?

WikiText-2, BPE-16000, dim 192, 4 layers, head_dim 32, kv_latent 48, seq 128,
500 steps, lr 3e-3. Three normalisers x two FFN types. softmax and sigmoid get
6 seeds; alpha-entmax gets 3 (it costs 2.0-2.3x and is the control).

Pairing verified at launch: within an FFN type all three arms share every weight
tensor bit-identically (46 tensors dense, 98 MoE); the only extras are the
per-head scalars `alpha_logit` / `gate_bias`.

| arm | n | mean ppl | sd | part_frac | sparsity |
|---|---|---|---|---|---|
| dense/softmax | 6 | 233.47 | 3.48 | 0.2682 | — |
| dense/sigmoid | 6 | 243.47 | 6.03 | 0.5526 | 0.2503 |
| dense/entmax | 3 | 238.75 | 8.36 | 0.1032 | 0.8059 |
| moe/softmax | 6 | 232.21 | 4.10 | 0.2882 | — |
| moe/sigmoid | 6 | 241.79 | 6.18 | 0.5273 | 0.3829 |
| moe/entmax | 3 | 233.42 | 6.57 | 0.1116 | 0.7862 |

## Result 1 — every sparsity mechanism is worse, and softmax wins outright

| paired contrast | n | mean | sd | t | crit | signs |
|---|---|---|---|---|---|---|
| dense/sigmoid vs softmax | 6 | **+9.99** | 3.38 | **+7.25** | 2.571 | 0/6 better |
| moe/sigmoid vs softmax | 6 | **+9.58** | 7.66 | **+3.06** | 2.571 | 0/6 better |
| dense/entmax vs softmax | 3 | +6.93 | 10.79 | +1.11 | 4.303 | 0/3 |
| moe/entmax vs softmax | 3 | +1.25 | 9.32 | +0.23 | 4.303 | 2/3 |

Sigmoid gating is **resolved worse** in both FFN types — ~+9.8 perplexity, about
4%, with **12 of 12 individual seeds on the losing side**. entmax is directionally
worse but never resolved, for reasons in Result 3.

## Result 2 — P0 was wrong, and instructively so

I predicted sigmoid would beat entmax, reasoning that a mechanism whose gradient
never dies must beat one whose gradient does. **entmax came out better in both
FFN types** (+0.89 dense, +4.75 moe, neither resolved).

That reasoning was about *trainability* and ignored *representational cost*.
Removing the simplex removes competition between keys, and the competition is
load-bearing: softmax forces keys to trade off against one another, and without
that the attention mass dilutes. The measurement is `participation_frac` 0.53
for sigmoid against 0.27-0.29 for softmax — sigmoid drops 25-38% of keys outright
and *still* attends about twice as broadly over what remains.

**Sparsity and concentration are different axes.** No earlier experiment could
express this: softmax cannot produce near-zeros at all, and T2's `beta` could
only move concentration.

So the ReLU objection was correct about entmax's gradients — verified directly,
7 of 7 zeroed keys had gradient exactly 0 and 200 SGD steps of direct pressure
could not revive one — but "avoid the dead gradient" was not sufficient guidance
for choosing a replacement.

## Result 3 — hard sparsity costs consistency, not mean quality

The paired standard deviations:

| | sigmoid | entmax |
|---|---|---|
| dense | 3.38 | **10.79** |
| moe | 7.66 | **9.32** |

entmax's spread is 1.2-3.2x sigmoid's, on half the sample. Its individual dense
differences were +19.39, +0.25, +1.16 — one catastrophic run and two at parity.

This is what an absorbing state predicts: whether a run lands well depends on
which keys happen to die before the gradient vanishes, and that is seed-luck.
**The dead gradient appears to cost variance rather than average quality** —
entmax's mean is *better* than sigmoid's in both arms.

Offered as the most likely reading, not as established: 3 seeds, and comparing
standard deviations is statistically weak.

## Result 4 — P2 holds at 6/6: learned sparsity drifts toward DENSITY

alpha settled at 1.4838, 1.4732, 1.4755 (dense) and 1.4837, 1.4814, 1.4846
(moe) — every run below the 1.5 initialisation.

With T2's learned `beta` at 0.934x (flatter) and the 40-step probe at
1.500 -> 1.490, that is **six measurements across two independent mechanisms,
in two codebases, all showing autoregressive attention pushing back toward
density when given a knob.** It also matches Correia et al.'s finding that
decoder self-attention prefers denser attention than encoder self-attention.

This is now the most robust result in the whole line of work.

## Result 5 — P3 holds: no interaction with MoE routing

The sigmoid penalty is +9.99 dense and +9.58 MoE — indistinguishable. Attention
sparsity reduces the token-token graph and MoE routing reduces the token-expert
graph; nothing connects them, and nothing in the data suggests otherwise. MoE
load balance stayed healthy throughout (0.75-0.85, zero dead experts).

## The honest limit (P4, registered before the run)

Sparse attention's claimed mechanism is preventing attention **dispersion** at
long context (`arXiv:2506.16640`): non-informative tokens accumulate mass as `n`
grows, and exact zeros stop that accumulation. **At seq_len 128 there is almost
no dispersion to prevent**, so this experiment tested sparsity where its
mechanism cannot operate.

The result therefore reads: *at short context, every sparsity mechanism tested
costs perplexity, and every learnable sparsity knob is used to become less
sparse.* It does **not** read "learned sparsity does not work".

**The follow-up is the length axis, not more seeds**: paired softmax vs sigmoid
vs entmax at seq_len 128 / 256 / 512, testing whether the penalty *shrinks* with
length. That is a trend test like T3's head_dim sweep, and it is the only way to
distinguish "sparsity does not help" from "we tested it where it cannot help".
Attention cost scales as n^2, so seq 512 needs its own budget.

## Method notes

Two bugs caught before they reached a result:

1. `near_zero_frac` counted causally masked positions as dropped keys, reporting
   3.4605 for a quantity bounded in [0,1]. Fixed pre-launch.
2. Mid-run I noted sigmoid's sparsity apparently falling across seeds and offered
   two readings, committing to drop it if `gate_mass` explained it. It did:
   r = -0.845, p = 0.034, while `participation_frac` stayed flat. A fixed 1e-3
   threshold measures scale and reports it as structure. **Withdrawn.**

One claim retracted mid-run: after row 13 the data looked like perplexity rising
monotonically with sparsity. Row 14 killed it — 77% zeros for +0.25 perplexity,
against 83% zeros for +19.39. It had been explicitly hedged to one seed.

---

# Two-hour run — what the seven ideas look like when tested

`ALLOCATION.md` designs a ~63 CPU-hour suite. This is what was actually run in
two hours, and it is a different kind of thing: gates, structural audits and
metrology, chosen because they answer something decisive at a cost the window
allows. **No perplexity comparison here resolves** — at 1000 steps (0.43 epoch)
against T2's measured seed sd of 1.5–5.0, none can. Every number below is either
a ratio, a correlation, an exact identity, or a property of a graph.

| idea | what was run | verdict |
|---|---|---|
| 3 — bits/byte as metric | full T6 round trip | **licensed** — it is the compressed size, to 0.01% |
| 6 — compressibility topology | T11's graph half, no training | **interior optimum confirmed**, and the standard design is off the frontier |
| 2 — conditional depth | T8's causality audit, no training | **the leak is real**: 5.8% of selections at `c=0.5` |
| 7 — residual-only propagation | T12 Gate A | **premise holds but is not learned** — R² 0.93 trained vs 0.93 at init |
| 4 — surprisal routing | T9's oracle bracket | **the confound is large**: ρ = +0.11, overlap 0.56 against a chance floor of 0.50 |
| 1 — conditional width | T7's random-router control | see G2 below |
| 5 — MDL objective | nothing | needs `mdl.py` and training runs; not attempted, and not to be read as tested |

---

## Idea 3 — bits-per-byte is the compressed size (`coding.py`)

Witten–Neal–Cleary arithmetic coder, order-1 byte model fitted on wikitext2
train, 32 KB of held-out valid coded.

| model | analytic bits/byte | emitted bits/byte | gap | round trip |
|---|---|---|---|---|
| order-0 | 4.7700 | 4.7703 | 0.01% | byte-exact |
| order-1 | 3.4651 | 3.4651 | 0.00% | byte-exact |

All three checks pass: the round trip is exact, the emitted rate equals the
analytic cross-entropy, and the order-1 saving is the same measured either way
(1.3050 analytic, 1.3052 emitted). **The unit the rest of the suite is
denominated in is real.** The model is deliberately trivial — the claim under
test is about the coder and the arithmetic, and a trained network would confound
a coder bug with a model bug.

What this does *not* establish: that compression predicts capability. T6 licenses
a measurement, nothing more.

## Idea 6 — the compressibility/mixing frontier (`graphs.py`)

Every mask carries **exactly 2040 edges** (24.7% of the causal upper triangle),
so density cannot explain any difference. `reach@4` is the fraction of
(query, earlier key) pairs joined by a path of ≤4 hops — the dependencies a
4-layer model can structurally represent.

| mask | zlib bytes | reach@4 | spectral gap | clustering | degree CV |
|---|---|---|---|---|---|
| p=0 (pure window) | 87 | 0.7558 | 0.031 | 0.764 | 0.205 |
| **p=0.01** | **234** | **0.9965** | 0.042 | 0.738 | 0.205 |
| **p=0.03** | **382** | **0.9994** | 0.057 | 0.705 | 0.205 |
| p=0.1 | 649 | 0.9979 | 0.103 | 0.610 | 0.205 |
| p=0.3 | 919 | 0.9918 | 0.238 | 0.422 | 0.205 |
| p=1 (random) | 1075 | 0.7645 | 0.657 | 0.314 | 0.205 |
| window+global | 355 | 0.6548 | 0.164 | 0.799 | 0.205 |
| bigbird | 894 | 0.9979 | 0.329 | 0.489 | 0.269 |
| modular | 387 | 0.9621 | 0.009 | 0.911 | 0.584 |

**Q1 holds**: compressed size rises monotonically with `p`, 87 → 1075 bytes.
Randomness is incompressible, as it must be.

**Q2 holds, and more sharply than predicted.** Reachability is *non-monotone*:
0.756 at `p=0`, 0.999 at `p=0.03`, back to 0.765 at `p=1`. The interior optimum
the small-world literature predicts is present, and it is cheap — `p=0.01` buys
99.7% reachability for 234 bytes, a quarter of the random graph's description
length. The mechanism is visible in the two ends: a pure lattice has no
shortcuts, and a purely random causal graph has destroyed the local chain that
carried information between *nearby* tokens, so both fail at 4 hops for opposite
reasons.

**Q3 holds**: `bigbird` (reach 0.998 at 894 B) sits on the sweep curve near
`p=0.3`. Its random edges are the shortcuts; it is the same object under another
name.

**Q4 fails, and the follow-up shows I drew the wrong conclusion from it.**
`window+global` has the *worst* reachability of anything tested — 0.655, below
even the pure window, with 0.098 in the far band.

The first reading was that global tokens are a bad design. That is wrong, and a
one-minute control settles it: the **same number of hubs, spread through the
sequence instead of parked at positions 0–3**, reaches **1.000 in every band**
at 461 bytes.

| mask | zlib B | near | window | mid | far |
|---|---|---|---|---|---|
| window+global (hubs at 0–3) | 355 | 1.000 | 1.000 | 1.000 | **0.098** |
| hubs strided /16 | 461 | 1.000 | 1.000 | 1.000 | **1.000** |
| hubs strided /32 | 581 | 1.000 | 1.000 | 1.000 | **1.000** |
| dilated (powers of two) | 972 | 1.000 | 1.000 | 1.000 | 1.000 |

So it is **hub placement, not the hub idea** — and the mechanism is the same
causal asymmetry as before, applied to relays rather than to routes. A hub at
position `h` can only ever carry information about positions `≤ h`, because
everything it can read lies below it. Hubs at the start therefore shorten paths
to tokens that were already reachable and relay nothing. This is why the
Longformer/BigBird global-token construction does not port to a decoder as-is:
there, global tokens attend *bidirectionally* and genuinely relay; under a causal
mask that direction is gone.

Two corrections to the earlier write-up follow. `window+global` is not "the
standard design" — in a decoder it is the **attention-sink** pattern, whose job
is softmax stability, not reach. And it is not "dominated": a hub layout that
respects the causal asymmetry is the best mask measured here.

**A limit of the compressibility proxy, visible in the same table.** Strided hubs
and dilated offsets are algorithmically trivial — a stride is a handful of bits
to describe — yet zlib rates them at 461–972 bytes against 234 for the random
small-world graph, because a row-major bitmap scan does not see column-wise
periodicity. **zlib understates the compressibility of structured masks**, which
means the frontier as drawn is pessimistic exactly where a designer would want to
look. A description length that counted the generating program, not the bitmap,
would move these points sharply left.

Caveat that limits all of it: this is graph structure. Which point a task wants
is not tested, and the reachability metric assumes every edge carries
information equally, which attention does not.

## Idea 2 — the MoD causality leak is real (`mod_audit.py`)

Top-k over the sequence is a comparison *between* tokens, so a token's selection
can depend on tokens after it. Comparing the full-sequence selection against the
prefix-only selection a generation-time router could compute:

| capacity | disagreement | per layer |
|---|---|---|
| 0.125 | 4.44% | 4.3 / 4.1 / 4.2 / 5.2% |
| 0.25 | 5.57% | 4.3 / 5.1 / 6.1 / 6.8% |
| 0.50 | 5.83% | 4.9 / 5.5 / 6.5 / 6.4% |
| 0.75 | 5.19% | 5.4 / 4.8 / 4.9 / 5.7% |

**The registered prediction was wrong.** I expected disagreement to fall as
capacity rises, because at `c → 1` almost everything is selected under either
rule. It is roughly flat, peaking at `c=0.5` — which in hindsight is the obvious
shape: `c=0.5` is where the selection boundary passes through the densest part of
the score distribution, so it is maximally sensitive to which scores are in view.

At ~6% of positions changing, any MoD result reported without the
causal-predictor evaluation is inflated. That is not a large leak, but it is
larger than the effect sizes this suite is powered to detect, so for T8 it is
load-bearing rather than cosmetic.

## Idea 7 — layers are redundant, but not because they learned to be (`triage.py`, G1)

R² of a least-squares map from each block's residual stream to the next.

| seed | perplexity | mean R² | R² at init | middle-layers R² | per pair |
|---|---|---|---|---|---|
| 0 | 164.90 | 0.9316 | 0.9297 | 0.9801 | 0.782 / 0.986 / 0.974 / 0.984 |
| 1 | 170.17 | 0.9332 | 0.9289 | 0.9752 | 0.792 / 0.961 / 0.989 / 0.990 |

**Gate A passes on the level and fails on the interpretation.** Mean R² is 0.93,
far above the 0.5 kill threshold, and the middle layers are 0.98 — only ~2% of
the variance entering a middle block is new. So there is a large predictable
component to subtract, and T12 stage B is buildable.

But the initialisation column is the result. R² at init is **0.9297**, and after
1000 steps it is **0.9316** — training moved it by 0.2%. The redundancy is a
property of the residual architecture, not something the model learned: a
residual block is near-identity at init and stays close to it. That reframes
idea 7. Predictive coding's premise was that *learned* representations are
mutually predictable and the code wastes bits on it; here the predictability is
the skip connection, which already costs nothing to transmit. Removing it saves
description length that was not being paid.

The honest next step is therefore not stage B as designed. It is to ask whether
the ~2% that *is* new per middle layer is where all the work happens, which is a
different and cheaper question.

## Idea 4 — the epistemic/aleatoric confound is large (`triage.py`, G3)

Per held-out token: total surprisal under the full model, against *reducible*
surprisal (truncated-depth loss minus full-depth loss — what the last block
actually bought that token).

| seed | ρ(total, reducible) | top-half overlap | reducible share of surprisal |
|---|---|---|---|
| 0 | +0.0650 | 0.5440 | 1.44% |
| 1 | +0.1472 | 0.5782 | 0.95% |

Chance overlap is 0.50. **A router selecting the top half of tokens by total
surprisal picks essentially the same set as flipping a coin against the tokens
extra depth actually helps** — 0.56 against a 0.50 floor, at ρ ≈ +0.11.

This is the objection quantified, and it is worse than the framing suggested. The
two signals are not merely imperfectly aligned; they are close to unrelated. A
non-parametric router on entropy or surprisal would spend its budget on tokens
that are unpredictable *and stay unpredictable*.

The second column is the one that limits the whole idea though: **only ~1.2% of
total surprisal is removable by the last block at all.** At this scale nearly all
surprisal is aleatoric, so the ceiling on *any* depth-routing policy — oracle
included — is about 1% of the loss. That is a statement about a 4-layer model on
0.43 epochs, not about language models, and it is exactly the kind of claim that
should invert at scale. But within this suite it says T9's ordering of arms
matters less than whether there is anything to allocate.

## Idea 1 — a frozen random router is not worse than a learned one (`triage.py`, G2)

6 paired seeds, MoE `N=4`, `k=2` + 1 shared. Identical initialisation and data
order; the control's router is frozen at its random init, so the arms differ in
one thing only.

| seed | learned | frozen | diff (frozen − learned) | load_bal L/F | router entropy L/F |
|---|---|---|---|---|---|
| 0 | 163.10 | 162.10 | −1.00 | 0.833 / 0.937 | 0.552 / 0.969 |
| 1 | 162.82 | 161.79 | −1.03 | 0.790 / 0.916 | 0.506 / 0.969 |
| 2 | 164.94 | 163.80 | −1.14 | 0.799 / 0.934 | 0.529 / 0.970 |
| 3 | 165.58 | 165.64 | **+0.06** | 0.826 / 0.920 | 0.460 / 0.970 |
| 4 | 167.52 | 164.50 | −3.02 | 0.760 / 0.927 | 0.483 / 0.968 |
| 5 | 165.76 | 163.61 | −2.15 | 0.809 / 0.847 | 0.494 / 0.966 |

Mean −1.38 (−0.84%), sd 1.07, **t = −3.17 against a critical 2.571 at 5 df**.
Sign test: 5/6, **two-sided p = 0.219**.

**The two tests disagree, so by this suite's own rule (§4.3 of `ALLOCATION.md`)
the perplexity result is reported as UNRESOLVED.** The t is carried by four small
consistent differences plus one 3-point outlier; the sign test, which ignores
magnitude, sees a 5/6 split that six coin flips produce more than a fifth of the
time. Writing that rule in advance is the only reason it is being applied to a
result that would otherwise have read as a clean finding.

**What is resolved is the mechanism, and it is 6/6 with no overlap between the
arms.** The learned router sharpens — router entropy 0.46–0.55 against a frozen
0.97 — and pays for it in load balance, 0.76–0.83 against 0.85–0.94. Training
makes the router commit, and committing costs it the even spread that a random
projection gets for nothing. That is a clean, reproducible effect in exactly the
instrumentation T2 found survives this budget when perplexity does not.

### The mutual-information column is withdrawn

`expert_mi` was intended to carry this gate. It cannot, and the reason is
arithmetic, not the models: 1419 distinct BPE types over 16,256 samples gives a
plug-in (Miller–Madow) MI bias of **0.189 bits** against measured values of
0.35–0.60. A third to a half of every number in that column is estimator bias.

Worse, the bias is not equal across arms: it grows with the number of occupied
joint cells, and the frozen arm — which spreads tokens more evenly — occupies
more of them. **The apparent "frozen router carries more token information" is
confounded in precisely the direction observed**, so the column supports no
comparison at all and the runner's automatic verdict line, which reads it,
should be ignored for this run.

The fix is a permutation null (shuffle expert labels, recompute, subtract) and
an order of magnitude more eval samples. Both are cheap; neither was in place,
and reporting the column as-is would have been the third result in this project
to be a measurement artefact wearing the costume of a finding.

### What it does to the suite

T7's P7.2 said MoE must beat a random router or the effect is capacity, not
conditional computation. On this evidence the learned router does not beat it,
and may lose to it. That is not yet a result — 1000 steps, 6 seeds, `N=4`,
and the two tests disagree — but it is the outcome P7.2 was written to catch,
and it lands on the premise T8 and T9 both rest on.

The re-scoping the gate calls for: before T8 and T9 ask *which* routing signal is
best, they must establish that **any** learned routing beats a fixed random
assignment at matched budget. That is a cheaper experiment than either, and it is
now the first thing in the DAG after T6.

## Idea 5 — MDL as the objective — NOT ATTEMPTED

Needs `mdl.py` (two-part accountant, hard-concrete L0 gates) and its own training
runs. Nothing about it was measured, and nothing above bears on it.

---

## What the two hours cost, and what it bought

Compute was not the constraint — the four completed pieces used ~10 CPU-minutes
between them, plus ~25 minutes for the two dense training runs. **Implementation
was the constraint**, which is the argument for the shared-infrastructure section
of `ALLOCATION.md` §2: `coding.py` and `graphs.py` were written here as
one-offs and are now the modules T6 and T11 need.

Three of the seven ideas turned out to be answerable — decisively — with no
training at all, because their load-bearing claims are about a coder, a graph, or
a selection rule rather than about a model. That is worth carrying back into the
full design: **T6 and the graph half of T11 should move to the front of the DAG
and stop being budgeted as GPU work.**

Two registered predictions failed (T8's capacity trend, T11's Q4). Both failures
were more informative than the corresponding passes, and neither would have been
visible from the design document.
