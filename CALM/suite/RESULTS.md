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

# T2-redux — was the temperature inert, or just redundant?

WikiText-2, BPE-16000, dim 192, 4 layers, head_dim 32, kv_latent 48, seq 128,
dense FFN, **500 steps, lr 3e-3, 4 seeds**, all four arms paired (25 of 25
shared tensors bit-identical at each seed; the only extra parameter is
`log_beta`). Script: `t2r_qknorm.py`, predictions registered in its docstring.

T2 found a learnable per-head `beta` worth nothing and settling at 0.934x.
`PHYSICS.md` sec.5.1 argued that experiment could not have concluded otherwise,
because `hybrid.py` feeds `q` and `k` straight from their projections: the
effective inverse temperature is `|q|*|k|*beta`, so `beta` was a redundant copy
of a knob the weights already had. This run closes that route with QK-norm —
RMS on `q` and `k` per head, no learnable gain, pinning `|q||k| = head_dim`.

| arm | n | mean ppl | sd | beta_x | qk_gain | part_frac | entropy |
|---|---|---|---|---|---|---|---|
| raw/fixed | 4 | 232.80 | 3.92 | 1.000 | 51.840 | 0.2640 | 0.5230 |
| raw/learned | 4 | 234.19 | 3.85 | 0.986 | 52.595 | 0.2729 | 0.5228 |
| qknorm/fixed | 4 | 236.63 | 3.86 | 1.000 | 32.000 | 0.2729 | 0.7297 |
| qknorm/learned | 4 | 239.48 | 11.23 | 1.110 | 32.000 | 0.2692 | 0.6975 |

The four `raw/fixed` rows reproduce T5's dense/softmax column to the digit
(228.26 / 230.81 / 236.37 / 235.76), so the harness is verified against an
independent earlier run before any of this is read.

## P1 — does closing the magnitude route make beta move? **Yes, cleanly.**

| | beta_ratio | per seed |
|---|---|---|
| qk_norm off | 0.9860 ± 0.0085 | 0.9947, 0.9829, 0.9756, 0.9910 |
| qk_norm on | 1.1100 ± 0.0434 | 1.1394, 1.1436, 1.0496, 1.1077 |

Complete separation, in magnitude and in sign: every seed below 1.000 with the
gain free, every seed above it with the gain pinned, and the departure from
1.000 is **8x larger** (0.110 vs 0.014). The direction also inverts — free-gain
`beta` drifts slightly flatter, as T2 found; pinned-gain `beta` goes ~11%
*sharper*, which is the direction T2's original P2 predicted and could not see.

**T2's null on temperature is an artefact of the parameterisation.** `beta` is
not inert; it was a flat direction. Restoring it as the only route to logit
scale gives it a reproducible, seed-stable optimum away from `1/sqrt(d)`.

## P2 — does the q/k gain move to cancel beta? **No. Withdrawn.**

Paired `d(qk_gain)` with qk_norm off: **+0.755 ± 7.236, 2 of 4 positive**
(+0.392, −0.388, −7.281, +10.298). No signal, and the largest differences point
both ways. With qk_norm on it is exactly 0.0000 in all four seeds, which
verifies the instrument.

So the *conclusion* of `PHYSICS.md` sec.5.1 survives (P1) but the *mechanism* it
gave does not. Endpoint compensation is not what happens. The account the data
supports is the weaker, more ordinary one: a redundant direction receives no
consistent gradient, so `beta` simply never moves far; it does not need the
weights to chase it.

There is also a methodological finding here, and it is the useful part. The
paired spread of `qk_gain` (± 7.2) is **larger** than its across-seed spread
(sd ≈ 8 for the raw column: 55.6, 59.5, 51.8, 40.4). Pairing controls the
initialisation, but a learnable `beta` perturbs the whole trajectory, so an
endpoint quantity like the gain is not pinned the way perplexity is. Testing
compensation properly means logging the gain *along* training, not at the end.

## P3 — one operating point? **Only for participation, and only within a normaliser.**

`participation_frac` is 0.264–0.273 in all four arms — the T2 attractor
reading survives on that statistic. But `entropy_norm` splits by column: 0.52
raw against 0.70–0.73 with QK-norm, a 40% difference at the same participation.

T2 read the two as interchangeable views of one operating point. They are not.
The same effective number of attended tokens is reached with a visibly
different distribution shape, so "the model converges to its phase" is a claim
about whichever statistic is being quoted, and future work has to name it.

## P4 — perplexity. **Noise, as registered.**

Paired `d(ppl)`: +1.39 ± 2.18 (qk_norm off), +2.85 ± 7.87 (on). Nothing.

One unregistered effect *is* consistent, and it is the largest here: at fixed
`beta`, QK-norm costs perplexity in **4 of 4 seeds** — +3.58, +4.49, +2.51,
+4.73, mean **+3.83 ± 1.00** on identical initialisations. Pinning `|q||k| = 32`
holds the model in a flatter distribution (entropy 0.73 vs 0.52) than it reaches
when free to grow the gain to 40–60, and it pays for that. Note also that
`qknorm/learned` has by far the widest perplexity spread (sd 11.23, one seed at
254.31): a free temperature with no magnitude route is the least stable of the
four arms.

## What this changes

1. **T2's headline is withdrawn.** "A learnable temperature is worth nothing"
   was measured on a parameterisation in which it could not have been worth
   anything. The corrected statement: with the q/k magnitude free, `beta` is
   redundant and does not move; with it pinned, `beta` moves reproducibly to
   ~1.11x `1/sqrt(d)`.
2. **The T0 re-opening argument is alive again, weakly.** C2 wanted a learnable
   `beta` to test whether HELM's collapse was a temperature bug. T2 said that
   route was worthless; it is not — but it only works alongside QK-norm, and
   QK-norm itself costs ~3.8 perplexity here, so the test is more expensive than
   C2 assumed.
3. **Perplexity still cannot referee anything at this budget.** Every claim above
   rests on the order parameters, as T2's own method note predicted it would.
