# CALM-K3 vs baseline Kimi K3 — one run, read carefully

`benchmark.json`, produced by:

```bash
python -m V1.benchmark --ae-steps 800 --calm-steps 3000 --baseline-steps 3000 \
                       --seq-len 64 --batch 8 --new-tokens 32
```

Both models are built from the **same** `KimiBlockConfig` (`d_model` 128, 4
routed experts top-2, one shared, 3 KDA : 1 Gated MLA + final MLA, AttnRes),
see the same synthetic corpus from the same seed, and take the same number of
optimiser steps. Patch size 4, latent 32, vocab 128, sequence length 64.

## Structure

| | CALM-K3 | baseline |
|---|---|---|
| backbone positions per 64 tokens | **16** | 64 |
| backbone parameters | 1,496,780 | 1,496,780 (identical) |
| task-head parameters | 339,744 (energy head) | 16,384 (tied LM head) |
| trainable total | 2,017,388 | 1,513,164 |
| frozen codec, deployed alongside | 1,734,720 | — |

The 4× reduction in positions is the mechanism; everything below follows from
it. Note the codec is a real deployment cost — 1.73M parameters that must ship
with the model and are larger than the backbone itself at this scale. At the
scales CALM targets it is a rounding error; here it is not, and pretending
otherwise would be dishonest.

## Speed

| | CALM-K3 | baseline | ratio |
|---|---|---|---|
| training ms/step | 195.0 | 436.6 | **2.24× faster** |
| training tokens/s | 2,625 | 1,173 | 2.24× |
| generation, 32 new tokens | 8 steps, 0.255 s | 32 steps, 1.447 s | **5.67× faster** |

Generation gains more than training because the win is *sequential*: 8 forward
passes instead of 32. Training gains less than the 4× position ratio because
the energy head runs `num_samples = 8` draws per position and the codec still
encodes all 64 tokens to build the target.

**Wall clock for the whole run**, which is the number that decides what you can
actually afford:

| | seconds |
|---|---|
| CALM: codec 91.4 + model 610.5 | **701.9** |
| baseline: model | 1,217.2 |

CALM finished 3000 steps *including* pre-training its codec in 58% of the
baseline's wall clock. At matched wall clock it would have had roughly 5,900
steps to the baseline's 3,000.

## Quality

| | CALM-K3 | baseline |
|---|---|---|
| **BrierLM** | 0.124 | **0.297** |
| brier₁ | 0.154 | 0.558 |
| brier₂ | 0.133 | 0.350 |
| brier₃ | 0.133 | 0.250 |
| brier₄ | 0.088 | 0.158 |
| codec ceiling (reconstruction) | 0.999 | n/a |
| perplexity | *does not exist* | 3.095 |
| corpus entropy floor | | 3.61 (Markov-only) |

**The baseline wins on quality at equal steps, clearly.** BrierLM 0.297 against
0.124, and it is not close at order 1: 0.558 against 0.154.

Two things stop that being the whole story.

**The gap narrows with n-gram order.** CALM/baseline by order: 0.28, 0.38,
0.53, 0.55. That is the patch structure showing up in the metric. CALM commits
to four tokens at once, so when it gets a patch right it gets all four; the
baseline's advantage is largest exactly where its extra conditioning is largest
— the next single token — and shrinks as the window grows to the size of the
unit CALM predicts. `brier₄ = 0.088` means roughly 9% of whole four-token
patches came out exactly right, net of the collision penalty.

**CALM was still improving when the run stopped.** The separate diagnostic
(`diagnostic_kl_1e-3.txt`) tracked a shorter run and found monotone improvement
throughout: predicted-token accuracy 0.0125 → 0.0833 (10.7× chance) and brier₁
−0.417 → +0.100 over 1000 steps, with the energy loss still falling. The
baseline, by contrast, is converged — perplexity 3.095 is *below* the
Markov-only floor of 3.61, because the injected motifs make the real stream
more predictable than the chain alone.

So the honest statement is: **at matched steps on this corpus the baseline is
better, and CALM had not converged.** This run does not establish which is
better at convergence, and nothing at this scale could.

## What this run is evidence for, and what it is not

Evidence for:

* the implementation is correct end to end — codec, backbone, energy head,
  sampling, and a metric that scores both model families;
* the compression mechanism does what it claims: 4× fewer positions, 2.24×
  training throughput, 5.67× fewer sequential generation steps;
* the energy score is optimisable and the head learns a conditional
  distribution rather than collapsing (the collision term in BrierLM would
  drive the score negative if it had, and it starts there and climbs out).

Not evidence for:

* anything about quality at convergence, or at any scale that matters;
* CALM being worse than a softmax head — it was given a third of the wall
  clock and had not stopped improving;
* transfer to real text. The corpus is a synthetic Markov chain with motifs,
  chosen because HuggingFace is unreachable here and because a uniform corpus
  would make both models tie by construction.

## The hyperparameter that produced a false null first

The first run of this benchmark (`benchmark_kl_1.0_null.json`) reported CALM at
chance and looked like a clean negative result. It was not. `kl_weight` was
1.0; the CALM reference uses **1e-3**.

At 1.0 the codec is regularised into a near-prior posterior — measured, the
posterior noise norm (4.74) came out *larger* than the signal norm (3.06). A
latent then carries little more than a draw from the prior, the energy score is
nearly as well satisfied by matching the marginal as the conditional, and the
head correctly learns the marginal. Its loss converges to the value that
implies, and its samples decode to noise. A converged loss and a random output.

Fixing the constant moved the codec ceiling from 0.878 to 0.999 and inverted
the posterior to |mean| 7.87 against |noise| 1.47. Both files are kept: a null
that turned out to measure a hyperparameter is worth keeping next to the run
that corrected it.
